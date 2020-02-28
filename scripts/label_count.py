sample_num = 0
label_num = 0
with open('data/coco/coco_train.txt', 'r') as fr:
    for line in fr:
        temp = line.strip().split('\t')[-1].split(',')
        label_num += len(temp)
        sample_num += 1

print('coco average label number:', label_num / sample_num)


sample_num = 0
label_num = 0
with open('data/visual_genome/vg500_train.txt', 'r') as fr:
    for line in fr:
        temp = line.strip().split('\t')[-1].split(',')
        label_num += len(temp)
        sample_num += 1

print('visual_genome average label number:', label_num / sample_num)
