import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties


def draw_heatmap(matrix, num_labels, name):
    xLabel = list(range(num_labels))
    yLabel = list(range(num_labels))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_yticks(range(num_labels))
    # ax.set_yticklabels(yLabel)
    # ax.set_xticks(range(num_labels))
    # ax.set_xticklabels(xLabel)
    im = ax.imshow(matrix, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.savefig('tmp/{}_heatmap.jpg'.format(name))

def main(name):
    data_path = os.path.join('temp', name, 'train.txt')
    label_path = os.path.join('temp', name, 'label.txt')
    label_list = [line.strip() for line in open(label_path)]
    label2id = {l:i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    coocurrence_matrix = np.zeros((num_labels, num_labels))
    for line in open(data_path):
        temp = line.strip().split('\t')[-1].split(',')
        labelid_list = [label2id[t] for t in temp]
        for i in range(len(labelid_list)):
            for j in range(i+1, len(labelid_list)):
                x = labelid_list[i]
                y = labelid_list[j]
                coocurrence_matrix[x, y] += 1
                coocurrence_matrix[y, x] += 1
    temp = coocurrence_matrix / coocurrence_matrix.sum()
    draw_heatmap(coocurrence_matrix, num_labels, name)
    np.save(os.path.join('temp', name, 'label_coocurrence.npy'), coocurrence_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='coco')
    args = parser.parse_args()
    main(args.data)