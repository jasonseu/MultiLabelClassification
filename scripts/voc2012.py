# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-20
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from os.path import join
from xml.dom.minidom import parse


data_dir = 'data/VOC2012'
anno_dir = os.path.join(data_dir, 'Annotations')
image_dir = os.path.join(data_dir, 'JPEGImages')
target_dir = 'temp/voc2012'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

train_txt = os.path.join(data_dir, 'ImageSets/Main/train.txt')
train_imgIds = [t.strip() for t in open(train_txt)]

label_set = set()
train_data = []
for img_id in train_imgIds:
    xml_path = os.path.join(anno_dir, '{}.xml'.format(img_id))
    dom_tree = parse(xml_path)
    root = dom_tree.documentElement
    objects = root.getElementsByTagName('object')
    labels = set()
    for obj in objects:
        if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
            continue
        tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
        labels.add(tag)
        label_set.add(tag)
    image_path = os.path.join(image_dir, '{}.jpg'.format(img_id))
    if not os.path.exists(image_path):
        raise Exception('file {} not found!'.format(image_path))
    train_data.append('{}\t{}\n'.format(image_path, ','.join(list(labels))))

with open(os.path.join(target_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)
label_set = sorted(list(label_set))
with open(os.path.join(target_dir, 'label.txt'), 'w') as fw:
    for line in label_set:
        fw.write(line+'\n')


val_txt = os.path.join(data_dir, 'ImageSets/Main/val.txt')
val_imgIds = [t.strip() for t in open(val_txt)]

val_data = []
for img_id in val_imgIds:
    xml_path = os.path.join(anno_dir, '{}.xml'.format(img_id))
    dom_tree = parse(xml_path)
    root = dom_tree.documentElement
    objects = root.getElementsByTagName('object')
    labels = set()
    for obj in objects:
        if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
            continue
        tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
        labels.add(tag)
    image_path = os.path.join(image_dir, '{}.jpg'.format(img_id))
    if not os.path.exists(image_path):
        raise Exception('file {} not found!'.format(image_path))
    val_data.append('{}\t{}\n'.format(image_path, ','.join(list(labels))))

with open(os.path.join(target_dir, 'val.txt'), 'w') as fw:
    fw.writelines(val_data)