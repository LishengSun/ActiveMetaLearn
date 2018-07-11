import random
import matplotlib.pyplot as plt 
import numpy as np
import os
import sys
from os import path
from subprocess import call

# sys.path.append(path.abspath('../utils/'))
# from utils import *

from src.utils.standardize_matrix import standardize
from src.utils.matrix_io import *
from src.utils.plot_multi_bands import *
from src.utils.setup_cofirank import *
from src.strategies import Strategy
import operator
import json
import seaborn as sns; sns.set(color_codes=True)


class Best_so_far(object):
	"""
	we suppose the performance in M are accuracy types, i.e higher is better
	if it is not the case, please use -M
	"""

	def __init__(self, M_name, M_train, M_test, results_dir, select_strategy, CofiRank_setup=None, num_landmarks=0):
		"""
		for the moment, do leave 1 dataset out => M_test.shape=[1, total_num_algo]
		"""
		self.M_name = M_name
		self.M_test = M_test
		self.M_train = M_train
		self.Cofi = CofiRank_setup
		self.results_dir = results_dir
		

		self.num_landmarks = num_landmarks
		self.landmarks = []
		self.median_landmarks = []

		self.total_num_algo = M_train.shape[1]
		self.select_strategy = select_strategy
		self.selected_algo = {}
		self.to_select_algo = range(self.total_num_algo)
		self.best_so_far = []
		self.test_known_values = {} # knownvalues = landmarks + esmated algos, keys=algo_interdit, values=known_perf
		self.num_cofi_run = 0


	def _select_next_algorithm(self):
		self.strategy = Strategy(self.M_train, self.M_test, self.total_num_algo, self.to_select_algo, self.selected_algo)
		if self.select_strategy == 'random':
			method_to_call = getattr(self.strategy, 'random')
			result = method_to_call()
			print result
			self.next_algo = result
		return self.next_algo
			

	def _evaluate_next_algorithm(self):
		self.next_algo = self._select_next_algorithm()
		########## ACTUALLY TRAIN AND TEST ##############
		
		self.next_score = self.M_test[self.next_algo]
		return self.next_algo, self.next_score

	
	def run_experiment(self, plot=True):
		
		# iteration = 0
		# while iteration < self.total_num_algo:
		while len(self.best_so_far) < self.total_num_algo: #5:
			# print 
			# print 
			# print 
			# print 'iteration ', iteration
			self.next_algo, self.next_score = self._evaluate_next_algorithm()
			self.selected_algo[self.next_algo] = self.next_score
			# if not self.total_num_algo > self.M_train.shape[0]:
			if self.next_algo in self.to_select_algo:
				self.to_select_algo.remove(self.next_algo)
			if len(self.best_so_far)>0 and self.next_score <= self.best_so_far[-1]: # higher is better
				print 'My next algo is < best so far, I include again best_so_far[-1]: ', self.best_so_far[-1]
				self.best_so_far.append(self.best_so_far[-1])

			else:
				print 'My next score is the best so far, I include self.next_score: ', self.next_score
				self.best_so_far.append(self.next_score)
			# iteration += 1

		return self



if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_name", help="Name of meta dataset in DATASETS",
	                    type=str)
	parser.add_argument("--select_strategy", help="Strategy for selecting next algorithms, available choices are:\
		random, simple_rank_with_median, active_meta_learning_with_cofirank, \
		median_landmarks_with_1_cofirank",
	                    type=str)
	parser.add_argument("--global_norm", help="Whether to globally normalize the dataset",
	                    type=bool)

	parser.add_argument("--results_dir", help="Path to save results",
	                    type=str)
	
	args = parser.parse_args()
	M_name = args.dataset_name
	select_strategy = args.select_strategy
	global_norm = args.global_norm
	results_dir = args.results_dir

	data_dir = 'DATASETS'
	M = np.loadtxt(os.path.join(data_dir, '%s/%s.data'%(M_name, M_name)))
	if global_norm:
		M = (M-np.nanmean(M))/np.nanstd(M)

	CofiRank_dir = '/cofirank/' # this is the path in the 
	CofiRank_setup = {'dir': CofiRank_dir, \
		'config_dir':os.path.join(CofiRank_dir, 'config'), \
		'config_file_train': M_name+'_train.cfg', \
		'config_file_svd': M_name+'_svd.cfg', \
		'config_file_median': M_name+'_median.cfg', \
		'trainData_dir': os.path.join(CofiRank_dir, 'data/lisheng-data/train'), \
		'svdData_dir': os.path.join(CofiRank_dir, 'data/lisheng-data/svd'), \
		'medianData_dir': os.path.join(CofiRank_dir, 'data/lisheng-data/median'), \
		'trainData_reldir': 'data/lisheng-data/train', \
		'svdData_reldir': 'data/lisheng-data/svd', \
		'medianData_reldir': 'data/lisheng-data/median', \
		'trainOut_dir': os.path.join(CofiRank_dir, 'trainOut_lisheng'), \
		'svdOut_dir': os.path.join(CofiRank_dir, 'svdOut_lisheng'), \
		'medianOut_dir': os.path.join(CofiRank_dir, 'medianOut_lisheng'), \
		'trainOut_reldir': 'trainOut_lisheng/', \
		'svdOut_reldir': 'svdOut_lisheng/', \
		'medianOut_reldir': 'medianOut_lisheng/', \
		'num_landmark': 3, \
		'cofidimW': 10,
		}

	################### leave 1 dataset out ###################
	best_so_far_alltest = {}
	for test_i in range(M.shape[0]):
		print '############################ima looking at test ', test_i
		M_train = np.copy(M)
		M_train = np.delete(M_train, test_i, axis=0)
		M_test = M[test_i, :]
		Exp = Best_so_far(M_name, M_train, M_test, results_dir, select_strategy, CofiRank_setup=None, num_landmarks=0)
		Exp.run_experiment()
		best_so_far_alltest['test_'+str(test_i)] = Exp.best_so_far
		json.dump(best_so_far_alltest, open(os.path.join(Exp.results_dir, \
			Exp.M_name+'_%s_best_so_far_alltest.txt'%select_strategy),'w')) # always save results













