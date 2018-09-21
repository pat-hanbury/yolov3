#darknet.py

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# define some layer classes

class EmptyLayer(nn.Module): # doesn't do much
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module): # defines some anchors
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_cfg(cfgfile):
    """
    imports cfg file

    returns lists of dictionaries of network blocks
    """
    file = open(cfgfile,'r')
    lines = file.read().split('\n') #get list of lines
    #keep lines if not blank and not commented
    lines = [x for x in lines if len(x) >0 and x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines] #clean up whitespace

    block = {} #dict for each block
    blocks = [] #list of blocks

    for line in lines:
        if line[0] == '[':
            if len(block) != 0: #if not the first time
                blocks.append(block) #add  previous block to list of blocks
                block = {} #reset
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    """inputs list of block dictionaries
        creates a list of modules to use in archetecture
    """
    net_info = blocks[0] #this is where preprocessing info is stored
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    #create a module for each block as we go
    for index, x in enumerate(blocks[1:]):

        module = nn.Sequential()
        if x["type"] == "convolutional":
            #get info about layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2

            # conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias,)
            module.add_module("conv_{0}".format(index),conv) # add conv_i to nn.Sequential list

            # add batch norm if applicable
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            # Check activation. If "leaky' it means add a leaky ReLu
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1,inplace=True)

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        elif x["type"] == "route":
            x["layers"] = x["layers"].split(',')
            # start of route
            start = int(x["layers"][0])
            # end if it exists
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x["type"] == "shortcut": # this keeps integrety of list
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo Detection layer
        elif x["type"] == "yolo":
            # mask specifies what anchors to use (0,1,2) and anchors has a list of all possible anchor tuples
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors) # where do we get this from?
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module) #add moduel of nn.Sequential to the module list
        prev_filters = filters # for next round
        output_filters.append(filters)

    return (net_info, module_list)


# define the network
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA=False):
        modules = self.blocks[1:]
        outputs = {} # contains output feature maps of each layer index : feature map

        # loop over modules (which is in same  order as module lists) and apply action
        write = 0
                for i, module in enumerate(modules):
                    module_type = module["type"]

                    if module_type == "convolutional" or module_type == "upsample":
                        x = self.module_list[i](x)

                    elif module_type == "route":
                        layers = module["layers"]
                        layers = [int(a) for a in layers]
                        if layers[0] > 0:
                             layers[0] = layers[0] - i

                        if len(layers) == 1
                            x = outputs[i + layers[0]] # does this just backtrack through layers?
                                # why???

                        else:
                            if layers[1] > 0:
                                layers[1] = layers[1] - i

                            # get two output feature maps
                            map1 = outputs[i + layers[0]]
                            map2 = outputs[i + layers[1]]

                            # concatanate along channel dimension
                            x = torch.cat((map1,map2), 1)

                    elif module_type == "shortcut":
                        from_var = int(module["from"])
                        # just add two layers?
                        x = outputs[i-1] + outputs[i+from_var]




