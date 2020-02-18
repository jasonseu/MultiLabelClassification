import os
import argparse
import numpy as np

# generate adjacency matrix
def preprocessing_for_ssgrl(dataset, fullname):
    dir_name = f'data/{fullname}'

    label_path = os.path.join(dir_name, 'label.txt')
    train_path = os.path.join(dir_name, f'{dataset}_train.txt')
    val_path = os.path.join(dir_name, f'{dataset}_val.txt')
    graph_path = os.path.join(dir_name, f'{dataset}_graph.npy')
    glove_path = '/nfs/share/CV_data/glove.840B.300d.txt'
    embed_path = os.path.join(dir_name, f'{dataset}_embeddings.npy')

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

    coco_embeddings = []
    for cat in categories:
        # category (eg: traffic light) with two or more words should split and average in each word embedding
        temp = np.array([list(map(lambda x: float(x), embeddings[t].split())) for t in cat.split()])   
        if temp.shape[0] > 1:
            temp = temp.mean(axis=0, keepdims=True)
        coco_embeddings.append(temp[0])

    coco_embeddings = np.array(coco_embeddings)
    np.save(embed_path, coco_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True')
    args = parser.parse_args()

    name_mapping = {
        'vg500': 'visual_genome',
        'coco': 'coco',
        'oi': 'open_images',
        'in': 'imagenet',
        'tc': 'tencent'
    }
    preprocessing_for_ssgrl(args.dataset, name_mapping[args.dataset])