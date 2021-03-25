# MultiLabelClassification

This is a multi label classification codebase in PyTorch. Currently, it supports ResNet101, SSGRL (a implement of paper "Learning Semantic-Specific Graph Representation for Multi-Label Image Recognition" based on official repository [HCPLab-SYSU/SSGRL](https://github.com/HCPLab-SYSU/SSGRL)) and training on CoCo, Visual Genome and Open Image.

## Requirements
- Python 3.6
- PyTorch 1.1
- TorchVision 0.3

## Pretrained models
Pretrained models are provided on [google drive](https://drive.google.com/open?id=10Ex1hEWCZw8Gop0DN-kvnPVlVfuzTbll). 

## Results
Typically, The performances of pretrained multi label classification models are evaluated with mean average precision (mAP) and reported as follows:

|   models  |  VOC2012  |  CoCo   |   VG500  |
|   -----   |  -------  |  -----  | ---------|
| ResNet101 |   0.901   |  0.802  |   0.293  |
| SSGRL     |           |  0.837  |   0.334  |

## Training

```bash
python train.py --config $cfg_file_path
```
For example, with default optimizer(Adam) and loss(BCElogitloss), training resnet101 model on different dataset: 
```bash
python train.py --config configs/coco_resnet101.yaml
python train.py --config configs/voc2012_resnet101.yaml
```
training ssgrl model on different dataset:
```bash
python train.py --config configs/coco_ssgrl.yaml
python train.py --config configs/voc2012_ssgrl.yaml
```

To resume training, you can run `train.py` with argument `--resume`.

## Evaluation

```bash
python evaluate.py --config $cfg_file_path
```
For example:
```bash
python evaluate.py --config configs/vg500_resnet101.yaml
python evaluate.py --config configs/vg500_ssgrl.yaml
```

## Inference

## Acknowledgements
Thanks the official implement [SSGRL](https://github.com/HCPLab-SYSU/SSGRL) and awesome PyTorch team.
