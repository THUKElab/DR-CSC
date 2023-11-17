"""
@Time   :   2021-01-12 15:10:43
@File   :   utils.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import json
import os
import sys

def write_low_mid_labels(results,low_labels,mid_labels,tokenizer):
    pred_lbl_pro_list = []
    for idx, (src_t_p,low_label,mid_label) in enumerate(zip(results,low_labels,mid_labels)):
        mid_label = mid_label.tolist()
        low_label = low_label.tolist()
        low_label.pop()
        mid_label.pop()
        low_label.pop(0)
        mid_label.pop(0)
        src, tgt, predict = src_t_p
        item = [str(idx)]
        for i, (a,b,c,d) in enumerate(zip(src,predict,low_label,mid_label), start=1):
            if c == 1:
                item.append(str(i))
                if d == 1:
                    item.append(tokenizer.convert_ids_to_tokens(a))
                    item.append('形')
                else:
                    item.append(tokenizer.convert_ids_to_tokens(a))
                    item.append('音')
        if len(item) == 1:
            item.append('0')
        pred_lbl_pro = ', '.join(item)
        pred_lbl_pro_list.append(pred_lbl_pro)
    return pred_lbl_pro_list

def compute_corrector_prf(results):
    """
    copy from https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
    """
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results:
        src, tgt, predict = item
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if detection_precision + detection_recall == 0:
        detection_f1 = 0
    else:
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
    print("The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall,
                                                                             detection_f1))

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results[i][1][j] in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if correction_precision + correction_recall == 0:
        correction_f1 = 0
    else:
        correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall)
    print("The correction result is precision={}, recall={} and F1={}".format(correction_precision,
                                                                              correction_recall,
                                                                              correction_f1))

    return detection_f1, correction_f1


def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)


def dump_json(obj, fp):
    try:
        fp = os.path.abspath(fp)
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        print(f'json文件保存成功，{fp}')
        return True
    except Exception as e:
        print(f'json文件{obj}保存失败, {e}')
        return False


def get_main_dir():
    # 如果是使用pyinstaller打包后的执行文件，则定位到执行文件所在目录
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    # 其他情况则定位至项目根目录
    return os.path.join(os.path.dirname(__file__), '..')


def get_abs_path(*name):
    return os.path.abspath(os.path.join(get_main_dir(), *name))

def compute_sentence_level_correction_realise(results):
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for item in results:
        src, tgt, predict = item
        if tgt != src: #原句有错
            targ_p += 1
        if predict != src: #预测改变了原句
            pred_p += 1
        if predict == tgt: #预测结果等于tgt
            hit += 1
        if predict != src and predict == tgt: #预测改变了原句且预测结果等于tgt
            tp += 1
    
    acc = hit / len(results)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    print(f'Sentence Level corredtion (Realise): acc:{acc:.6f}, precision:{p:.6f}, recall:{r:.6f}, f1:{f1:.6f}')
    return acc, p, r, f1

def compute_sentence_level_detection_realise(results):
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for item in results:
        src, tgt, predict = item
        tgt_label = []
        pred_label = []
        for idx in range(len(src)):
            if src[idx] != tgt[idx]:
                tgt_label.append((idx, tgt[idx]))
            if src[idx] != predict[idx]:
                pred_label.append((idx, predict[idx]))
        if tgt != src: #原句有错
            targ_p += 1
        if predict != src: #预测改变了原句
            pred_p += 1
        if len(tgt_label) == len(pred_label) and all(p[0] == t[0] for p,t in zip(pred_label,tgt_label)):
            hit += 1
        if predict != src and len(tgt_label) == len(pred_label) and all(p[0] == t[0] for p,t in zip(pred_label,tgt_label)):
            tp += 1
    
    acc = hit / len(results)
    p = tp / pred_p if tp > 0 else 0.0
    r = tp / targ_p if tp > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    print(f'Sentence Level detection (Realise): acc:{acc:.6f}, precision:{p:.6f}, recall:{r:.6f}, f1:{f1:.6f}')
    return acc, p, r, f1

def compute_sentence_level_detection(results):
    tp, tn, fp, fn, total = 0, 0, 0, 0, 0
    for item in results:
        src, tgt, predict = item
        tgt_label = []
        pred_label = []
        for idx in range(len(src)):
            if src[idx] != tgt[idx]:
                tgt_label.append((idx, tgt[idx]))
            if src[idx] != predict[idx]:
                pred_label.append((idx, predict[idx]))
        if src == tgt:
            if len(pred_label) == len(tgt_label) and all(p[0] == t[0] for p,t in zip(pred_label,tgt_label)):
                tn += 1
            else:
                fp += 1
        else:
            if len(pred_label) == len(tgt_label) and all(p[0] == t[0] for p,t in zip(pred_label,tgt_label)):
                tp += 1
            else:
                fn += 1
    acc = (tp + tn) / len(results)
    p = tp / (tp + fp) if tp > 0 else 0.0
    r = tp / (tp + fn) if tp > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    print(f'Sentence Level detection (Raw): acc:{acc:.6f}, precision:{p:.6f}, recall:{r:.6f}, f1:{f1:.6f}')
    return acc, p, r, f1
        

def compute_sentence_level_prf(results):
    """
    自定义的句级prf，设定需要纠错为正样本，无需纠错为负样本
    :param results:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = len(results)

    for item in results:
        src, tgt, predict = item

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == predict:
                TN += 1
            # 预测为正
            else:
                FP += 1
        # 正样本
        else:
            # 预测也为正
            if tgt == predict:
                TP += 1
            # 预测为负
            else:
                FN += 1

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    print(f'Sentence Level correction (Raw): acc:{acc:.6f}, precision:{precision:.6f}, recall:{recall:.6f}, f1:{f1:.6f}')
    return acc, precision, recall, f1
