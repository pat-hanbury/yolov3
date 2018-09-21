#utils.py

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
    bs = prediction.size(0)
    stride = inp_dim // prediction.size(2) # inp wid/prediction wid
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(bs, bbox_attrs*num_anchors, grid_size*grid_size)

    anchors = [(a[0]/stride , a[1]/stride) for a in anchors] #anchors pixel dim are relative
    # to original input image dimensions, so they have to be adjusted for stride

    # we need to transform our bounding box outputs using the equations given in the paper
    # center x, center y and object confidence score is passed through sig function
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 0])

    # Add center offsets
    grid = np.arrange(grid_size)