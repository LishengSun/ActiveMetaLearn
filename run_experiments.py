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
		method_to_call = getattr(self.strategy, self.select_strategy)
		self.next_algo = method_to_call()
		return self.next_algo
			

	def _evaluate_next_algorithm(self):
		"""
		might be separated and more complicated later
		"""

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
	parser.add_argument("-d", "--dataset_name", help="Name of meta dataset in DATASETS",
	                    type=str, required=True)
	parser.add_argument("-s", "--select_strategies", help="Strategies for selecting next algorithms, available choices are:\
		random, simple_rank_with_median, active_meta_learning_with_cofirank, \
		median_landmarks_with_1_cofirank. You can choose one or more strategies to compare",
	                    type=str, nargs='+', required=True)
	parser.add_argument("-n", "--global_norm", help="Whether to globally normalize the dataset, default is True",
	                    default=True, type=bool)

	parser.add_argument("-rd", "--results_dir", help="Path to save results, default is the current dir",
	                    default=os.getcwd(), type=str)
	
	args = parser.parse_args()
	M_name = args.dataset_name
	select_strategies = args.select_strategies
	global_norm = args.global_norm
	results_dir = args.results_dir

	if not os.path.exists(results_dir): # create results_dir if not existed
		os.makedirs(results_dir)
    

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
	################### plot setting ############################
	fig, ax = plt.subplots()
	plot_colors = {'random':'blue', 'simple_rank_with_median': 'green', \
	'active_meta_learning_with_cofirank': 'red', 'median_landmarks_with_1_cofirank':'magenta'}
	plot_marker = {'random': 's', 'simple_rank_with_median': '<', \
	'active_meta_learning_with_cofirank': 's', 'median_landmarks_with_1_cofirank':'o'}
	plot_makersize = {'random': 4, 'simple_rank_with_median': 8, \
	'active_meta_learning_with_cofirank': 12, 'median_landmarks_with_1_cofirank':10}
	plot_label = {'random': 'Random: median', 'simple_rank_with_median': 'SimpleRank w. median', \
	'active_meta_learning_with_cofirank': 'Active_Meta_Learning w. CofiRank', \
	'median_landmarks_with_1_cofirank': 'Median_LandMarks w. 1-CofiRank'}
	# plot_markerfacecolor = {'random': 'none', 'simple_rank_with_median': 'none', 'active_meta_learning_with_cofirank':'none', }
	plot_markeredgecolor = {'random': 'none', 'simple_rank_with_median': 'none', \
	'active_meta_learning_with_cofirank':'red', 'median_landmarks_with_1_cofirank':'magenta'}
	plot_markeredgewidth = {'random': None, 'simple_rank_with_median': None, \
	'active_meta_learning_with_cofirank':2, 'median_landmarks_with_1_cofirank': 2}
	plot_ymin = []

	random_run_num = 10 # statistics of random is based on 1000 iterations

	for stra in select_strategies:
		print '================== STRA: %s ===================='%stra
		best_so_far_alltest = {}
		for test_i in range(M.shape[0]):
			print '############################ima looking at test ', test_i
			M_train = np.copy(M)
			M_train = np.delete(M_train, test_i, axis=0)
			M_test = M[test_i, :]
			if stra == 'random':
				best_so_far_random_test_i = []
				for run in range(random_run_num):
					Exp = Best_so_far(M_name, M_train, M_test, results_dir, stra, CofiRank_setup=None, num_landmarks=0)
					Exp.run_experiment()
					best_so_far_random_test_i.append(Exp.best_so_far)
				best_so_far_alltest['test_'+str(test_i)] = best_so_far_random_test_i
				
			else:
				Exp = Best_so_far(M_name, M_train, M_test, results_dir, stra, CofiRank_setup=CofiRank_setup, num_landmarks=3)
				Exp.run_experiment()
				best_so_far_alltest['test_'+str(test_i)] = Exp.best_so_far

			json.dump(best_so_far_alltest, open(os.path.join(Exp.results_dir, \
				Exp.M_name+'_%s_best_so_far_alltest.txt'%stra),'w')) # always save results

		num_ds = len(best_so_far_alltest)
		arr = np.array([best_so_far_alltest['test_'+str(i)] for i in range(num_ds)])
		
		if stra == 'random':
			##### compute percentiles ########
			random_Q5 = []
			random_Q25 = []
			random_Q75 = []
			random_Q95 = []
			random_Median = []
			for ds_i in range(arr.shape[0]):
				# print '==================random_Q5', 
				# print random_Q5
				random_Q5_i = np.nanpercentile(arr[ds_i,:,:], 5, axis=0)
				random_Q25_i = np.nanpercentile(arr[ds_i,:,:], 25, axis=0)
				random_Median_i = np.nanpercentile(arr[ds_i,:,:], 50, axis=0)
				random_Q75_i = np.nanpercentile(arr[ds_i,:,:], 75, axis=0)
				random_Q95_i = np.nanpercentile(arr[ds_i,:,:], 95, axis=0)
				random_Q5.append(random_Q5_i)
				random_Q25.append(random_Q25_i)
				random_Median.append(random_Median_i)
				random_Q75.append(random_Q75_i)
				random_Q95.append(random_Q95_i)
			random_Q5 = np.array(random_Q5).reshape(len(random_Q5),-1)	
			random_Q25 = np.array(random_Q25).reshape(len(random_Q25),-1)	
			random_Median = np.array(random_Median).reshape(len(random_Median),-1)	
			random_Q75 = np.array(random_Q75).reshape(len(random_Q75),-1)	
			random_Q95 = np.array(random_Q95).reshape(len(random_Q95),-1)
			random_Q5 = np.nanmean(random_Q5, axis=0)	
			random_Q25 = np.nanmean(random_Q25, axis=0)	
			random_Median = np.nanmean(random_Median, axis=0)
			random_Q75 = np.nanmean(random_Q75, axis=0)	
			random_Q95 = np.nanmean(random_Q95, axis=0)	

		
			ax.plot(range(1,M.shape[1]+1), random_Median, color=plot_colors[stra], marker=plot_marker[stra],\
			markeredgecolor=plot_markeredgecolor[stra], markeredgewidth=plot_markeredgewidth[stra], \
			markersize=plot_makersize[stra], label=plot_label[stra])

			ax.fill_between(range(1,M.shape[1]+1), random_Q5, random_Q25, alpha=.15, facecolor="blue", edgecolor="none", label='Random: 5-25% quantiles')
			ax.fill_between(range(1,M.shape[1]+1), random_Q25, random_Median, alpha=.2, facecolor="blue", edgecolor="none", label='Random: 25-50%')
			ax.fill_between(range(1,M.shape[1]+1), random_Median, random_Q75, alpha=.25, facecolor="blue", edgecolor="none", label='Random: 50-75%')
			ax.fill_between(range(1,M.shape[1]+1), random_Q75, random_Q95, alpha=.3, facecolor="blue", edgecolor="none", label='Random: 75-95%')
			plot_ymin += list(random_Median)
			
		else:
			ax.plot(range(1,M.shape[1]+1), np.nanmean(arr, axis=0), color=plot_colors[stra], marker=plot_marker[stra],\
			markeredgecolor=plot_markeredgecolor[stra], markeredgewidth=plot_markeredgewidth[stra], \
			markersize=plot_makersize[stra], label=plot_label[stra])
			plot_ymin += list(np.mean(arr, axis=0))

	ax.set_xlabel('number of algorithms estimated so far', fontsize=15)
	ax.set_ylabel('best performance so far', fontsize=15)
	ymin = np.nanmin(plot_ymin)

	ax.set_ylim(bottom=ymin-0.1)
	
	ax.set_title(M_name)
	plt.savefig(os.path.join(results_dir, M_name+'_AVE_%i_r1000'%M.shape[0]))
	plt.show()











