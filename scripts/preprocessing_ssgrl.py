# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-3-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import argparse
import numpy as np


glove_path = '/nfs/users/zhuxuelin/public/glove.840B.300d.txt'

# generate adjacency matrix
def preprocessing_for_ssgrl(data):
    dir_name = os.path.join('temp', data)

    label_path = os.path.join(dir_name, 'label.txt')
    train_path = os.path.join(dir_name, 'train.txt')
    val_path = os.path.join(dir_name, 'val.txt')
    graph_path = os.path.join(dir_name, 'graph.npy')
    embed_path = os.path.join(dir_name, 'embeddings.npy')

    categories = [line.strip() for line in open(label_path).readlines()]
    cate2id = {cat:i for i, cat in enumerate(categories)}
    adjacency_matrix = np.zeros((len(categories), len(categories)))

    with open(train_path, 'r') as fr:
        data = [line.strip().split('\t')[1].split(',') for line in fr.readlines()]
    with open(val_path, 'r') as fr:
        data.extend([line.strip().split('\t')[1].split(',') for line in fr.readlines()])

    for temp in data:
        for i in temp:
            for j in temp:
                adjacency_matrix[cate2id[i], cate2id[j]] += 1

    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix[i] = adjacency_matrix[i] / adjacency_matrix[i, i]
        adjacency_matrix[i, i] = 0.0

    np.save(graph_path, adjacency_matrix)

    # generate coco category embeddings
    with open(glove_path, 'r') as fr:
        embeddings = dict([line.split(' ', 1) for line in fr.readlines()])

    data_embeddings = []
    for cat in categories:
        if cat == 'diningtable': # pretrained glove missing the label diningtable in voc2012
            cat = 'dining table'
        if cat == 'tvmonitor':
            cat = 'tv monitor'
        if cat == 'pottedplant':
            cat = 'potted plant'
        # category (eg: traffic light) with two or more words should split and average in each word embedding
        temp = np.array([list(map(lambda x: float(x), embeddings[t].split())) for t in cat.split()])   
        if temp.shape[0] > 1:
            temp = temp.mean(axis=0, keepdims=True)
        data_embeddings.append(temp[0])

    data_embeddings = np.array(data_embeddings)
    np.save(embed_path, data_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['coco', 'voc2012', 'vg500'])
    args = parser.parse_args()

    preprocessing_for_ssgrl(args.data)