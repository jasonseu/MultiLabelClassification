# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-19
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import json
from collections import defaultdict

coco_dir = 'data/coco'
save_dir = 'temp/coco'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = json.load(open(os.path.join(coco_dir, 'annotations/instances_train2014.json')))
categories = []
catId2catName = {}
for line in data['categories']:
    catId2catName[line['id']] = line['name']
    categories.append(line['name'])
imgId2imgName = {}
for line in data['images']:
    imgId2imgName[line['id']] = line['file_name']

imgName2catName = defaultdict(list)
for line in data['annotations']:
    cat_id = line['category_id']
    cat_name = catId2catName[cat_id]
    img_id = line['image_id']
    img_name = imgId2imgName[img_id]
    img_path = os.path.join(coco_dir, 'train2014', img_name)
    if cat_name not in imgName2catName[img_path]:
        imgName2catName[img_path].append(cat_name)

train_data = ['{}\t{}\n'.format(k, ','.join(v)) for k, v in imgName2catName.items()]
print(f"total training data number: {len(train_data)}")

with open(os.path.join(save_dir, 'label.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(x) for x in categories])
with open(os.path.join(save_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)


data = json.load(open(os.path.join(coco_dir, 'annotations/instances_val2014.json')))
catId2catName = {}
for line in data['categories']:
    catId2catName[line['id']] = line['name']
imgId2imgName = {}
for line in data['images']:
    imgId2imgName[line['id']] = line['file_name']

imgName2catName = defaultdict(list)
for line in data['annotations']:
    cat_id = line['category_id']
    cat_name = catId2catName[cat_id]
    img_id = line['image_id']
    img_name = imgId2imgName[img_id]
    img_path = os.path.join(coco_dir, 'val2014', img_name)
    if cat_name not in imgName2catName[img_path]:
        imgName2catName[img_path].append(cat_name)

imgName2catName = ['{}\t{}\n'.format(k, ','.join(v)) for k, v in imgName2catName.items()]
print(f"total test data number: {len(imgName2catName)}")

with open(os.path.join(save_dir, 'val.txt'), 'w') as fw:
    fw.writelines(imgName2catName)
