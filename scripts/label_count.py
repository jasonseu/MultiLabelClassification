label_num = {}
with open('temp/voc2012/train.txt', 'r') as fr:
    for line in fr:
        temp = line.strip().split('\t')[-1].split(',')
        for t in temp:
            if t in label_num.keys():
                label_num[t] += 1
            else:
                label_num[t] = 1

print('voc2012 average label number:', label_num)


# sample_num = 0
# label_num = 0
# with open('temp/coco/train.txt', 'r') as fr:
#     for line in fr:
#         temp = line.strip().split('\t')[-1].split(',')
#         label_num += len(temp)
#         sample_num += 1

# print('coco average label number:', label_num / sample_num)


# sample_num = 0
# label_num = 0
# with open('temp/visual_genome/vg500_train.txt', 'r') as fr:
#     for line in fr:
#         temp = line.strip().split('\t')[-1].split(',')
#         label_num += len(temp)
#         sample_num += 1

# print('visual_genome average label number:', label_num / sample_num)
