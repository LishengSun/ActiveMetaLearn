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

	@staticmethod
	def projection_between_vectors_nan(vec1, vec2):
		"""
		vec1 * vec2 = sum(vec1[notnan]*vec2[notnan])
		"""
		proj = 0
		for ele1 in vec1:
			if not np.isnan(ele1):
				for ele2 in vec2:
					if not np.isnan(ele2):
						proj += abs(ele1*ele2)
		return proj

	@staticmethod
	def select_landmark_CofiRank(U, M, num_landmark):
		"""
		M~U * V, then M[:,j] = sum(col_wight_c * U[:,c])
		"""
		col_weight = []
		for M_col in range(M.shape[1]): # for each column of M_train, test its projection to U's columns
			weight = 0
			for U_col in range(U.shape[1]):
				projection_to_U = projection_between_vectors_nan(M[:,M_col], U[:,U_col])

				weight += projection_to_U
			col_weight.append(weight)
		landmarks = sorted(range(len(col_weight)), key=lambda k: col_weight[k], reverse=True)[:num_landmark]
		return landmarks, col_weight


	@staticmethod
	def select_landmark_SVD(M, num_landmark):
		"""
		select landmarks as columns most projected to V's first columns
		where V satisfies M = USV (from svd)
		here M is assumed to be full
		"""
		
		landmarks = []
		U, s, V = np.linalg.svd(M)
		pcomp = U[:num_landmark,:]
		for pc_idx in range(len(pcomp)):
			col_weight_pc = []
			for M_col in range(M.shape[1]): # for each column of M_train, test its projection to U's columns
				cw = abs(np.inner(M[:,M_col], pcomp[pc_idx]))
				col_weight_pc.append(cw)
			for col in sorted(range(len(col_weight_pc)), key=lambda k: col_weight_pc[k], reverse=True):
				if not col in landmarks:
					landmarks.append(col)
		return landmarks#, col_weight



	def random(self):
		self.next_algo = random.choice(self.to_select_algo)
		return self.next_algo


	def simple_rank_with_median(self):
		self.median_rank = list(reversed(np.argsort(np.nanmedian(self.M_train, axis=0))))
		self.next_algo = self.median_rank[len(self.selected_algo)] # selected_algo tracks the num_iteration
		return self.next_algo





