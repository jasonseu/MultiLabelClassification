from models.resnet101 import ResNet101
from models.ssgrl import SSGRL

model_factory = {
    'resnet101': ResNet101,
    'ssgrl': SSGRL
}