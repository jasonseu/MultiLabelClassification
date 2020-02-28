# MultiLabelClassification

This is a multi label classification codebase in PyTorch. Currently, it supports ResNet101, SSGRL (a implement of paper "Learning Semantic-Specific Graph Representation for Multi-Label Image Recognition" based on official repository [HCPLab-SYSU/SSGRL](https://github.com/HCPLab-SYSU/SSGRL)) and training on CoCo, Visual Genome and Open Image.

## Requirements
- Python 3.6
- PyTorch 1.1
- TorchVision 0.3

## Pretrained models
Pretrained models are provided on [google drive](https://drive.google.com/open?id=10Ex1hEWCZw8Gop0DN-kvnPVlVfuzTbll). 

## Results
Typically, The performances of pretrained multi label classification models are evaluated with mean average precision (mAP) and displayed as follows:

|   models  |  CoCo   | Visual Genome | Open Image V5 |
|   -----   |  -----  | ------------- | ------------- |
| ResNet101 | 0.801   |   0.250       |      -        |
| SSGRL     | 0.836   |   0.294       |      -        |

## Training

```bash
python train.py --dataset $dataset --model $model --loss $loss --optimizer $optimizer --batch_size $batch_size
```
For example, with default optimizer(Adam) and loss(BCElogitloss), training resnet101 model on different dataset: 
```bash
python train.py --dataset coco --model resnet101 --batch_size 32
python train.py --dataset vg500 --model resnet101 --batch_size 32
```
training ssgrl model on different dataset:
```bash
python train.py --dataset coco --model ssgrl --batch_size 8
python train.py --dataset vg500 --model ssgrl --batch_size 8
```

To resume training, you can run `train.py` with argument `--resume`.

## Testing

```bash
python test.py --dataset $dataset --model $model --batch_size $batch_size
```
For example:
```bash
python test.py --dataset vg500 --model resnet101 --batch_size 32
python test.py --dataset coco --model resnet101 --batch_size 32
python test.py --dataset vg500 --model ssgrl --batch_size 8
python test.py --dataset coco --model ssgrl --batch_size 8
```

## Inference

## Acknowledgements
Thanks the official implement [SSGRL](https://github.com/HCPLab-SYSU/SSGRL) and awesome PyTorch team.
