import os
import json
from random import shuffle
from collections import defaultdict

coco_dir = 'data/coco'

data = json.load(open(os.path.join(coco_dir, 'instances_train2014.json')))
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
    if cat_name not in imgName2catName[img_name]:
        imgName2catName[img_name].append(cat_name)

imgName2catName = ['{}\t{}\n'.format(k, ','.join(v)) for k, v in imgName2catName.items()]
shuffle(imgName2catName)
train_data = imgName2catName[:len(imgName2catName)-5000]
validation_data = imgName2catName[len(imgName2catName)-5000:]

print(f"total training data number: {len(train_data)}")
print(f"total validation data number: {len(validation_data)}")

with open(os.path.join(coco_dir, 'label.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(x) for x in categories])
with open(os.path.join(coco_dir, 'coco_train.txt'), 'w') as fw:
    fw.writelines(train_data)
with open(os.path.join(coco_dir, 'coco_val.txt'), 'w') as fw:
    fw.writelines(validation_data)


data = json.load(open(os.path.join(coco_dir, 'instances_val2014.json')))
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
    if cat_name not in imgName2catName[img_name]:
        imgName2catName[img_name].append(cat_name)

imgName2catName = ['{}\t{}\n'.format(k, ','.join(v)) for k, v in imgName2catName.items()]
print(f"total test data number: {len(imgName2catName)}")

with open(os.path.join(coco_dir, 'coco_test.txt'), 'w') as fw:
    fw.writelines(imgName2catName)
