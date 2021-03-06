from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy

torch.set_grad_enabled(False);
import itertools
import seaborn as sns

import panopticapi
from panopticapi.utils import id2rgb, rgb2id


CLASSES = [
     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
     'Backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
     'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
     'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
     'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
     'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
     'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
    if c != "N/A":
        coco2d2[i] = count
        count+=1
# print(coco2d2)

transform = T.Compose([
     T.Resize(800),
     T.ToTensor(),
     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model.eval()

url = "http://images.cocodataset.org/val2017/000000281759.jpg"
im = Image.open(requests.get(url, stream=True).raw)

img = transform(im).unsqueeze(0)
out = model(img)


print(out['pred_logits'].shape)

# scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]

# exit()