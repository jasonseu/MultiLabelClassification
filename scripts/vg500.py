# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-19
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import json
import random
from random import shuffle

from collections import Counter

random.seed(202)

target_dir = 'temp/vg500'
image_dir1 = 'data/VisualGenome1.4/VG_100K'
image_dir2 = 'data/VisualGenome1.4/VG_100K_2'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

data = json.load(open('data/VisualGenome1.4/objects.json'))
print('total image number: {}'.format(len(data)))

tags = [temp['names'][0] for line in data for temp in line['objects']]
counter = Counter()
counter.update(tags)
print('total tag number: {}'.format(len(tags)))
print('total unique tag number: {}'.format(len(counter)))

tags500 = [k for k, _ in counter.most_common()[:500]]

vg500_dict = []
for line in data:
    temp = {}
    temp['image_id'] = line['image_id']
    temp['objects'] = []
    for obj in line['objects']:
        if obj['names'][0] in tags500 and obj['names'][0] not in temp['objects']:
            temp['objects'].append(obj['names'][0])
    if len(temp['objects']) > 0:
        vg500_dict.append(temp)

vg500 = []
for item in vg500_dict:
    labels = ','.join(item['objects'])
    img_name = '{}.jpg'.format(item['image_id'])
    img_path = os.path.join(image_dir1, img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(image_dir2, img_name)
        if not os.path.exists(img_path):
            raise Exception('file {} not found!'.format(img_path))
    vg500.append('{}\t{}\n'.format(img_path, labels))

shuffle(vg500)
train_num = int(len(vg500) * 0.8)
train_split = vg500[:train_num]
test_split = vg500[train_num:]

print('total number of train dataset: {}'.format(len(train_split)))
# # print('total number of validation dataset: {}'.format(len(validation_split)))
print('total number of test dataset: {}'.format(len(test_split)))

with open(os.path.join(target_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_split)
# with open('data/vg500/val.txt', 'w') as fw:
#     fw.writelines(validation_split)
with open(os.path.join(target_dir, 'val.txt'), 'w') as fw:
    fw.writelines(test_split)
with open(os.path.join(target_dir, 'label.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in tags500])
