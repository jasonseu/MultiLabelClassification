# MultiLabelClassification

This is a multi label classification codebase in PyTorch. Currently, it supports ResNet101, SSGRL (a implement of paper "Learning Semantic-Specific Graph Representation for Multi-Label Image Recognition" based on official repository [HCPLab-SYSU/SSGRL](https://github.com/HCPLab-SYSU/SSGRL)) and training on Pascal Voc 2012, COCO and Visual Genome.

## Requirements
- Python 3.6
- PyTorch 1.1
- TorchVision 0.3

## Data preparation
Download datasets and symlink the paths to them as follows:
```bash
mkdir data
ln -s /path/to/mscoco data/coco
ln -s /path/to/VisualGenome1.4 data/VisualGenome1.4
ln -s /path/to/VOC2012 data/VOC2012

mkdir tmp
ln -s /path/to/glove.840B.300d.txt tmp/glove.840B.300d.txt
```

Running following scripts to preprocess datasets and generate desired data for SSGRL model.
```
python scripts/voc2012.py
python scripts/coco.py
python scripts/vg500.py

python scripts/preprocessing_ssgrl.py --data [voc2012, coco, vg500]
```

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

## Pretrained models
Pretrained models are provided on [google drive](https://drive.google.com/open?id=10Ex1hEWCZw8Gop0DN-kvnPVlVfuzTbll). 

## Evaluation

```bash
python evaluate.py --config $cfg_file_path
```
For example:
```bash
python evaluate.py --config configs/vg500_resnet101.yaml
python evaluate.py --config configs/vg500_ssgrl.yaml
```

## Results
Typically, The performances of pretrained multi label classification models are evaluated with mean average precision (mAP) and reported as follows:

|   models  |  VOC2012  |  COCO   |   VG500  |
|   -----   |  -------  |  -----  | ---------|
| ResNet101 |   0.901   |  0.802  |   0.293  |
| SSGRL     |   0.923   |  0.837  |   0.334  |

## Acknowledgements
Thanks the official implement [SSGRL](https://github.com/HCPLab-SYSU/SSGRL) and awesome PyTorch team.
