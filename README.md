# DR-CSC
The repository of EMNLP 2023 "A Frustratingly Easy Plug-and-Play Detection-and-Reasoning Module for  Chinese Spelling Check"

## Data Preparation
1. download the training data from [https://cloud.tsinghua.edu.cn/d/576a2219778847dea3f4/](https://cloud.tsinghua.edu.cn/d/576a2219778847dea3f4/)
2. put the training data in the data/
3. check the files are in data/
   ```
   confusionset_xin.txt
   confusionset_yin.txt
   test_13_with_mid.json
   test_14_with_mid.json
   test_15_with_mid.json
   train_with_mid.json
   ```

## Bert model Preparation
1. download the chinese-bert-wwm model from [https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) or the model also can be found [https://cloud.tsinghua.edu.cn/d/576a2219778847dea3f4/](https://cloud.tsinghua.edu.cn/d/576a2219778847dea3f4/)

## Train
```
python -u main.py --mode train \
    --gpu_index 0 \
    --sighan 15 \
    --lr=5e-5 \
    --epochs=20 \
    --batch_size=32 \
    --accumulate_grad_batches=2 \
    --model_save_path output/pro_15 \
    --model pro \
    --load_checkpoint f \
    --without_M \
    --loss_weight 0.5
```

## Test
```
python main.py --mode test \
    --gpu_index 0 \
    --sighan 15 \
    --model_save_path output/pro_15 \
    --model pro \
```