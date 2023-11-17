"""
@Time   :   2021-01-12 15:08:01
@File   :   models.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import operator
import os
from collections import OrderedDict

import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertOnlyMLMHead
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
import transformers

from .utils import compute_corrector_prf, compute_sentence_level_prf, compute_sentence_level_correction_realise,\
                    compute_sentence_level_detection_realise, compute_sentence_level_detection,\
                    write_low_mid_labels
import numpy as np


class DetectionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.config.hidden_size, 1)

    def forward(self, hidden_states):
        out, _ = self.gru(hidden_states)
        prob = self.linear(out)
        prob = self.sigmoid(prob)
        return prob


class BertCorrectionModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(self.config)
        self.corrector = BertEncoder(self.config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pooler = BertPooler(self.config)
        self.cls = BertOnlyMLMHead(self.config)
        self._device = device

    def forward(self, texts, prob, embed=None, cor_labels=None, residual_connection=False):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self._device)
            # torch的cross entropy loss 会忽略-100的label
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        if embed is None:
            embed = self.embeddings(input_ids=encoded_texts['input_ids'],
                                    token_type_ids=encoded_texts['token_type_ids'])
        # 此处较原文有一定改动，做此改动意在完整保留type_ids及position_ids的embedding。
        # mask_embed = self.embeddings(torch.ones_like(prob.squeeze(-1)).long() * self.mask_token_id).detach()
        # 此处为原文实现
        mask_embed = self.embeddings(torch.tensor([[self.mask_token_id]], device=self._device)).detach()
        cor_embed = prob * mask_embed + (1 - prob) * embed

        input_shape = encoded_texts['input_ids'].size()
        device = encoded_texts['input_ids'].device

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(encoded_texts['attention_mask'],
                                                                                 input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.corrector(
            cor_embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        sequence_output = sequence_output + embed if residual_connection else sequence_output
        prediction_scores = self.cls(sequence_output)
        out = (prediction_scores, sequence_output, pooled_output)

        # Masked language modeling softmax layer
        if text_labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='sum')  # -100 index = padding token
            cor_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), text_labels.view(-1))
            out = (cor_loss,) + out
        return out

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = torch.load(gen_fp)
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
            if name.startswith('encoder'):
                name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)

from transformers import BertTokenizer
class BaseCorrectorTrainingModel(pl.LightningModule):
    """
    用于CSC的BaseModel, 定义了训练及预测步骤
    """

    def __init__(self, arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = arguments
        self.w = arguments.loss_weight
        self.min_loss = float('inf')
        self.max_cf = 0
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_checkpoint)

    def training_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels,mid_label,_ = batch
        outputs,_,_ = self.forward(ori_text, cor_text, det_labels, mid_label=mid_label, is_train=True)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        return loss

    def validation_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels, mid_label,transfer_data = batch
        outputs,low_labels,mid_labels = self.forward(ori_text, cor_text, det_labels, mid_label=mid_label,transfer_data = transfer_data)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_text, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        if 13 == self.args.sighan:
            cor_y_hat[(cor_y_hat == self.tokenizer.convert_tokens_to_ids('地')) | (cor_y_hat == self.tokenizer.convert_tokens_to_ids('得'))] = \
                encoded_x['input_ids'][(cor_y_hat == self.tokenizer.convert_tokens_to_ids('地')) | (cor_y_hat == self.tokenizer.convert_tokens_to_ids('得'))]

        results = []
        det_acc_labels = []
        cor_acc_labels = []
        for src, tgt, predict, det_predict, det_label in zip(ori_text, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
            cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
            det_acc_labels.append(det_predict[1:len(_src) + 1].equal(det_label[1:len(_src) + 1]))
            results.append((_src, _tgt, _predict,))
        if self.args.model == 'pro':
            return loss.cpu().item(), det_acc_labels, cor_acc_labels, results,low_labels.cpu(), mid_labels.cpu()
        else:
            return loss.cpu().item(), det_acc_labels, cor_acc_labels, results,[], []

    def on_validation_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        print('Valid.')

    def validation_epoch_end(self, outputs) -> None:
        det_acc_labels = []
        cor_acc_labels = []
        results = []
        low_labels, mid_labels = [],[]
        for out in outputs:
            det_acc_labels += out[1]
            cor_acc_labels += out[2]
            results += out[3]
            low_labels += out[4]
            mid_labels += out[5]
        loss = np.mean([out[0] for out in outputs])
        print(f'loss: {loss}')
        print(f'Detection:\n'
              f'acc: {np.mean(det_acc_labels):.4f}')
        print(f'Correction:\n'
              f'acc: {np.mean(cor_acc_labels):.4f}')
        print('Char Level:')
        # if '13' in self.args.label_file:
        #     predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
        #         input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
        compute_corrector_prf(results)
        compute_sentence_level_detection(results)
        compute_sentence_level_prf(results)
        compute_sentence_level_detection_realise(results)
        acc, p, r, f1 = compute_sentence_level_correction_realise(results)
        if self.args.model == 'pro':
            pred_lbl_pro_list = write_low_mid_labels(results,low_labels,mid_labels,self.tokenizer)
        if self.args.transfer_path is None and self.args.model == 'pro':
            label_pro_path = os.path.join(self.args.model_save_path,'smb_labels_pro.txt')
            with open(label_pro_path,'w',encoding='utf-8') as f:
                f.write('\n'.join(pred_lbl_pro_list))
        # if (len(outputs) > 5) and (loss < self.min_loss):
        if (len(outputs) > 5) and (f1 > self.max_cf):
            # self.min_loss = loss
            self.max_cf = f1
            torch.save(self.state_dict(),
                       os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))
            print('model saved.')
        torch.save(self.state_dict(),
                   os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        print('Test.')
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = LambdaLR(optimizer,
                             lr_lambda=lambda step: min((step + 1) ** -0.5,
                                                        (step + 1) * self.args.warmup_epochs ** (-1.5)),
                             last_epoch=-1)
        return [optimizer], [scheduler]


class SoftMaskedBertModel(BaseCorrectorTrainingModel):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.args = args
        self.config = BertConfig.from_pretrained(args.bert_checkpoint)
        self.detector = DetectionNetwork(self.config)
        self.tokenizer = tokenizer
        self.corrector = BertCorrectionModel(self.config, tokenizer, args.device)
        self._device = args.device

    def forward(self, texts, cor_labels=None, det_labels=None,mid_label=None, is_train=False,transfer_data=None):
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        embed = self.corrector.embeddings(input_ids=encoded_texts['input_ids'],
                                          token_type_ids=encoded_texts['token_type_ids'])
        prob = self.detector(embed)
        cor_out = self.corrector(texts, prob, embed, cor_labels, residual_connection=True)
        # corr_out[0] loss
        # corr_out[1] prediction_scores [B,L,Vocab]
        # corr_out[2] hidden [B,L,hidden]
        # corr_out[3] after pooling [B,L]

        if det_labels is not None:
            det_loss_fct = nn.BCELoss(reduction='sum')
            # pad部分不计算损失
            active_loss = encoded_texts['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            outputs = (det_loss, cor_out[0], prob.squeeze(-1)) + cor_out[1:]
            # outputs[0] det_loss
            # outputs[1] cor_loss
            # outputs[2] det
            # outputs[3] prediction_scores [B,L,Vocab]
            # outputs[4] hidden [B,L,hidden]
            # outputs[5] after pooling [B,L] 
        else:
            outputs = (prob.squeeze(-1),) + cor_out

        return outputs,None,None
    
    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.corrector.load_from_transformers_state_dict(gen_fp)

class Progressive_MultiTaskHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform_low = BertPredictionHeadTransform(config)
        self.low_level = nn.Linear(config.hidden_size, 2)

        self.transform_mid = BertPredictionHeadTransform(config) 
        self.mid_level = nn.Linear(config.hidden_size, 2)
        # self.high_level = BertLMPredictionHead(config)
    
    def forward(self, low_sequence_output, mid_sequence_output):
        sequence_output_low = self.transform_low(low_sequence_output)
        sequence_output_mid = self.transform_mid(mid_sequence_output)
        # low_level
        low_logits = self.low_level(sequence_output_low)

        # mid_level
        mid_logits = self.mid_level(sequence_output_mid)

        # high_level
        # high_logits = self.high_level(sequence_output)

        # return low_logits, mid_logits, high_logits
        return low_logits, mid_logits    


class SoftMaskedBertModelPro(BaseCorrectorTrainingModel):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.args = args
        self.config = BertConfig.from_pretrained(args.bert_checkpoint)
        self.config.without_M = args.without_M
        self.config.low_ground_truth = args.low_ground_truth
        self.config.mid_ground_truth = args.mid_ground_truth
        self.detector = DetectionNetwork(self.config)
        self.tokenizer = tokenizer
        self.corrector = BertCorrectionModel(self.config, tokenizer, args.device)
        self._device = args.device
        self.prog_multi_task = Progressive_MultiTaskHeads(self.config)
        self.confusionset_xin = self.build_confusionset('data/confusionset_xin.txt', self.tokenizer)
        self.confusionset_yin = self.build_confusionset('data/confusionset_yin.txt', self.tokenizer)

    def both_in_confusion(self,tmp_word, tmp_label, typee):
        if typee == 'xin':
            return tmp_word in self.confusionset_xin.keys() and tmp_label in self.confusionset_xin[tmp_word]
        if typee == 'yin':
            return tmp_word in self.confusionset_yin.keys() and tmp_label in self.confusionset_yin[tmp_word]

    def build_confusionset(self, confusionset_path, tokenizer):
        confusionset = {}
        tokenizer = tokenizer
        with open(confusionset_path ,"r",encoding="utf-8") as file:
            for line in file:
                w, confusion = line.strip().split(":")
                w = tokenizer.convert_tokens_to_ids(w)
                if w == tokenizer.unk_token_id:
                    continue
                confusion = tokenizer.convert_tokens_to_ids(list(confusion))
                confusionset[w] = list(confusion)
        return confusionset

    def forward(self, texts, cor_labels=None, det_labels=None, mid_label=None, is_train=False, transfer_data=None):
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        embed = self.corrector.embeddings(input_ids=encoded_texts['input_ids'],
                                          token_type_ids=encoded_texts['token_type_ids'])
        prob = self.detector(embed)
        cor_out = self.corrector(texts, prob, embed, cor_labels, residual_connection=True)

        batch_size, seq_len, vocab_size = cor_out[1].shape[0],cor_out[1].shape[1],cor_out[1].shape[2]
        if mid_label is not None:
            # max_length = max([len(i) for i in mid_label])
            # mid_label = list(mid_label)
            # for each_mid_label in mid_label:
            #     each_mid_label.insert(0,0)
            #     if seq_len - max_length > 1:
            #         each_mid_label.append(0)
            mid_label = [torch.LongTensor(i) for i in mid_label]
            mid_label = nn.utils.rnn.pad_sequence(mid_label, batch_first=True, padding_value=1)
        encoded_x = cor_out[2]
        low_logits, mid_logits = self.prog_multi_task(encoded_x, encoded_x)

        if is_train:
            low_labels = (~torch.eq(encoded_texts['input_ids'].cpu(),
                                    encoded_labels['input_ids'])).long()
            mid_labels = mid_label
            mid_active_loss = ~torch.eq(encoded_texts['input_ids'].cpu(), 
                                        encoded_labels['input_ids']).view(-1)
        else:
            low_labels = torch.argmax(low_logits, dim = -1)
            mid_labels = torch.argmax(mid_logits, dim = -1)
            mid_active_loss = ~torch.eq(encoded_texts['input_ids'].cpu(), 
                                        encoded_labels['input_ids']).view(-1)
            if transfer_data is not None:
                        # transfer low,mid data from other model
                transfer_item = transfer_data
                transfer_low = torch.zeros_like(mid_labels[0])
                transfer_mid = torch.zeros_like(mid_labels[0])
                for i in range(1,len(transfer_item),3):
                    if transfer_item[i] == '0':
                        break
                    wrong_idx = int(transfer_item[i])
                    wrong_chr = transfer_item[i+1]
                    wrong_type = transfer_item[i+2]
                    transfer_low[wrong_idx] = 1
                    if wrong_type == '形':
                        transfer_mid[wrong_idx] = 1
                    else:
                        transfer_mid[wrong_idx] = 0
                low_labels = transfer_low.unsqueeze(0)
                mid_labels = transfer_mid.unsqueeze(0)
            if not self.config.without_M:
                if self.config.low_ground_truth:
                    low_labels = (~torch.eq(encoded_texts['input_ids'].cpu(),
                                    encoded_labels['input_ids'])).long()
                if self.config.mid_ground_truth:
                    mid_labels = mid_label
                mid_labels = mid_labels.cpu() * low_labels.cpu()
                is_wrong = torch.eq(low_labels, torch.ones_like(low_labels)) # 标记是否为错误的token，错误为True
                wrong_idx = torch.where(is_wrong == True) #错误token的坐标
                # index_to_select = input_ids.view(-1).cpu()
                # M_all = torch.index_select(self.confusiongraph,0, index_to_select).view(batch_size, seq_len, -1)
                M_all = torch.ones(batch_size, seq_len, vocab_size)
                for i,j in zip(wrong_idx[0].cpu(), wrong_idx[1].cpu()):
                    i = int(i)
                    j = int(j)
                    if mid_labels[i][j] == 1: #xin
                        confusionset = self.confusionset_xin
                    else:
                        confusionset = self.confusionset_yin
                    tmp_word = int(encoded_texts['input_ids'][i][j].cpu())
                    if tmp_word in confusionset.keys():
                        confusion_words = confusionset[tmp_word]
                        if tmp_word not in confusion_words: # confusion word需要包含自己
                            confusion_words = confusion_words + [tmp_word]
                        confusion_words_tensor = torch.tensor(confusion_words)
                        ones = torch.ones(len(confusion_words))
                        tmp_M = torch.zeros(vocab_size)
                        tmp_M.scatter_(0, confusion_words_tensor, ones)
                    else:
                        tmp_M = torch.ones(vocab_size)
                    M_all[i][j] = tmp_M
                M_all = M_all.detach()
                M_all = M_all.to(self._device)
        
        if not is_train and not self.config.without_M:
            cor_out = list(cor_out)
            cor_out[1] = cor_out[1].mul(M_all)
            cor_out = tuple(cor_out)
        
        if det_labels is not None:
            det_loss_fct = nn.BCELoss(reduction='sum')
            pro_loss_fct = nn.CrossEntropyLoss(reduction= 'none')
            # pad部分不计算损失
            active_loss = encoded_texts['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            # low level loss
            active_loss = encoded_texts['attention_mask'].view(-1) == 1
            active_labels = torch.where(active_loss,low_labels.view(-1).to(self._device), torch.tensor(pro_loss_fct.ignore_index).type_as(low_labels).to(self._device))
            low_loss = pro_loss_fct(low_logits.view(-1, 2), active_labels)
            # mid level loss
            active_labels = torch.where(
                mid_active_loss.to(self._device), mid_labels.view(-1).to(self._device), torch.tensor(pro_loss_fct.ignore_index).type_as(mid_labels).to(self._device)
            )
            mid_loss = pro_loss_fct(mid_logits.view(-1, 2), active_labels)
            def weighted_mean(weight, input):
                return torch.sum(weight * input) / torch.sum(weight)
            low_loss = weighted_mean(torch.ones_like(low_loss), low_loss)
            mid_loss = weighted_mean(torch.ones_like(mid_loss), mid_loss)
            pro_loss = low_loss + mid_loss
            outputs = (det_loss+pro_loss, cor_out[0], prob.squeeze(-1)) + cor_out[1:]
        else:
            outputs = (prob.squeeze(-1),) + cor_out

        return outputs,low_labels, mid_labels

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.corrector.load_from_transformers_state_dict(gen_fp)
