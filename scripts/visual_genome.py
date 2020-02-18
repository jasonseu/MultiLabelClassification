import json
from random import shuffle

from collections import Counter


data = json.load(open('data/visual_genome/objects.json'))
print('total image number: {}'.format(len(data)))

tags = [temp['names'][0] for line in data for temp in line['objects']]
counter = Counter()
counter.update(tags)
print('total tag number: {}'.format(len(tags)))
print('total unique tag number: {}'.format(len(counter)))

tags500 = [k for k, v in counter.most_common()[:500]]

vg500 = []
for line in data:
    temp = {}
    temp['image_id'] = line['image_id']
    temp['objects'] = []
    for obj in line['objects']:
        if obj['names'][0] in tags500 and obj['names'][0] not in temp['objects']:
            temp['objects'].append(obj['names'][0])
    if len(temp['objects']) > 0:
        vg500.append(temp)

vg500 = ['{}\t{}\n'.format(item['image_id'], ','.join(item['objects'])) for item in vg500]

shuffle(vg500)
train_num = int(len(vg500) * 0.8)
train_split = vg500[:train_num-5000]
validation_split = vg500[train_num-5000:train_num]
test_split = vg500[train_num:]

print('total number of train dataset: {}'.format(len(train_split)))
print('total number of validation dataset: {}'.format(len(validation_split)))
print('total number of test dataset: {}'.format(len(test_split)))

with open('data/visual_genome/vg500_train.txt', 'w') as fw:
    fw.writelines(train_split)
with open('data/visual_genome/vg500_val.txt', 'w') as fw:
    fw.writelines(validation_split)
with open('data/visual_genome/vg500_test.txt', 'w') as fw:
    fw.writelines(test_split)
with open('data/visual_genome/vg500_label.txt', 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in tags500])
