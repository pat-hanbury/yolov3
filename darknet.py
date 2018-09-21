#darknet.py

from __future__ import division

import toch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
		if line[0] == '['
			if len(block) != 0: #if not the first time
				blocks.append(block) #add  previous block to list of blocks
				block = {} #reset
			block["type"] = line[1:-1].rstrip()
		else:
			key,value = line.split("=")
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)
	
	return block
	
	
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
		
		if (x["type"]) == "convolutional"):
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