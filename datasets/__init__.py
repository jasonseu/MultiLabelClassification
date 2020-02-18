from datasets.visual_genome import VisualGenomeDataset
from datasets.open_images import OpenImagesDataset
from datasets.coco import CoCoDataset

data_factory = {
    'vg500': VisualGenomeDataset,
    'oi': OpenImagesDataset,
    'coco': CoCoDataset
}