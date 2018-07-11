import random
import matplotlib.pyplot as plt 
import numpy as np
import os
import sys
from os import path
from subprocess import call

# sys.path.append(path.abspath('../utils/'))
from utils.standardize_matrix import standardize
from utils.matrix_io import *
from utils.plot_multi_bands import *
from utils.setup_cofirank import *
import operator
import json
import seaborn as sns; sns.set(color_codes=True)


class Strategy(object):

	def __init__(self, M_train, M_test, total_num_algo, to_select_algo, selected_algo):
		self.M_train = M_train
		self.M_test = M_test
		self.total_num_algo = total_num_algo
		self.to_select_algo = to_select_algo
		self.selected_algo = selected_algo

	def random(self):
		self.next_algo = random.choice(self.to_select_algo)
		return self.next_algo
