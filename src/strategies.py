import random
import matplotlib.pyplot as plt 
import numpy as np
import os
import sys
from os import path
from subprocess import call


########### set up logging ################
import logging
module_logger = logging.getLogger('run_experiments_logging.strategies')# create logger


# sys.path.append(path.abspath('../utils/'))
from utils.standardize_matrix import standardize
from utils.matrix_io import *
from utils.plot_multi_bands import *
from utils.setup_cofirank import *
import operator
import json
import seaborn as sns; sns.set(color_codes=True)


class Strategy(object):

	def __init__(self, M_name, M_train, M_test, total_num_algo, to_select_algo, selected_algo, \
		Cofi, num_cofi_run, current_rank, num_landmarks, median_landmarks):
		self.M_name = M_name
		self.M_train = M_train
		self.M_test = M_test
		self.total_num_algo = total_num_algo
		self.to_select_algo = to_select_algo
		self.selected_algo = selected_algo
		self.Cofi = Cofi
		self.num_cofi_run = num_cofi_run
		self.current_rank = current_rank
		self.num_landmarks = num_landmarks
		self.median_landmarks = median_landmarks


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
		self.current_rank = None
		return self.next_algo, self.num_cofi_run, self.current_rank, self.median_landmarks


	def simple_rank_with_median(self):
		self.median_rank = list(reversed(np.argsort(np.nanmedian(self.M_train, axis=0))))
		self.next_algo = self.median_rank[len(self.selected_algo)] # selected_algo tracks the num_iteration
		self.current_rank = self.median_rank
		return self.next_algo, self.num_cofi_run, self.current_rank, self.median_landmarks


	def simple_rank_with_svd(self):
		pass



	def active_meta_learning_with_cofirank(self):
		M_train_Cofi = np.vstack((self.M_train, self.M_test))
		module_logger.info('selected_algo: {}'.format(self.selected_algo.keys()))
		if self.selected_algo.keys() == []: # cold start
			M_train_Cofi[-1, :] = np.nanmedian(self.M_train, axis=0)
			module_logger.info('COLD START: I know nothing, use top 1 median: {} to initialize'.format(M_train_Cofi[-1, :]))
			rank_test = list(reversed(np.argsort(M_train_Cofi[-1, :])))
			module_logger.debug('median ranking: {}'.format(rank_test))
			self.next_algo = rank_test[0]
			module_logger.debug('self.next_algo = {}'.format(self.next_algo))
			
		elif len(self.selected_algo) == 2**(self.num_cofi_run+1) - 1: #run cofi only at 2**n iterations
			module_logger.info('Run cofi only at every 2**n iterations')
			module_logger.debug('=====================len(self.selected_algo) = %i, FIRE COFI'%len(self.selected_algo))
			self.num_cofi_run += 1
			module_logger.debug('self.selected_algo = {}'.format(self.selected_algo.keys()))
			for ind_col in range(len(M_train_Cofi[-1, :])):
				if ind_col in self.selected_algo.keys():
					M_train_Cofi[-1, ind_col] = self.M_test[ind_col]
				else:
					M_train_Cofi[-1, ind_col] = np.nan

			matrix_to_lsvm(M_train_Cofi, row_range=(range(M_train_Cofi.shape[0])), start_idx=0, \
				save_name=self.M_name+'_train.lsvm', save_dir=self.Cofi['trainData_dir'])
			M_test_Cofi = self.M_test

			matrix_to_lsvm(M_test_Cofi, row_range=(range(M_train_Cofi.shape[0])), start_idx=M_train_Cofi.shape[0]-1, \
				save_name=self.M_name+'_test.lsvm', save_dir=self.Cofi['trainData_dir'])
			# os.chdir(CofiRank_dir)
			os.chdir(self.Cofi['dir'])
			make_config_file(config_dir=self.Cofi['config_dir'], new_config_filename=self.Cofi['config_file_train'], \
				DtrainFile=os.path.join(self.Cofi['trainData_reldir'], self.M_name+'_train.lsvm'), \
				DtestFile=os.path.join(self.Cofi['trainData_reldir'], self.M_name+'_test.lsvm'), \
				outfolder=self.Cofi['trainOut_reldir'])
			call(['./dist/cofirank-deploy', 'config/%s'%self.Cofi['config_file_train']])
			F = lsvm_to_matrix(M_train_Cofi.shape, 'F.lsvm', lsvm_dir=self.Cofi['trainOut_dir'])
			F_test = F[-1, :]
			self.Cofi_rank_test = list(reversed(np.argsort(F_test)))
			module_logger.debug('Cofi ranking: {}'.format(self.Cofi_rank_test))
			self.current_rank = self.Cofi_rank_test
			# self.current_rank = list(reversed(np.argsort(F_test)))
			# module_logger.debug('Cofi ranking: {}'.format(self.current_rank))
			for top_alg in self.Cofi_rank_test:
				module_logger.debug('considering: {}'.format(top_alg))
				if not top_alg in self.selected_algo.keys():
					module_logger.debug('will choose as self.next_algo: {}'.format(top_alg))
					self.next_algo = top_alg
					break
			
			module_logger.debug('=================== number of Cofi run = {}'.format(self.num_cofi_run))
		else:
			module_logger.debug('=====================len(self.selected_algo) = %i != %i, NON NEW COFI RUN, still use last Cofi ranking'%(len(self.selected_algo), 2**(self.num_cofi_run+1)-1))
			module_logger.debug('number of Cofi run: {}'.format(self.num_cofi_run))
			for top_alg in self.current_rank:
				module_logger.debug('considering {}'.format(top_alg))
				if not top_alg in self.selected_algo.keys():
					module_logger.debug('will choose as self.next_algo: {}'.format(top_alg))
					self.next_algo = top_alg
					break
		
		return self.next_algo, self.num_cofi_run, self.current_rank, self.median_landmarks
		

	def	median_landmarks_with_1_cofirank(self):
		if self.median_landmarks == []:
			module_logger.info('self.median_landmarks == [], select top {} median cols as landmarks'.format(self.num_landmarks))
			self.median_rank = list(reversed(np.argsort(np.nanmedian(self.M_train, axis=0))))
			self.median_landmarks = self.median_rank[:self.num_landmarks]
			module_logger.debug('Now evaluate landmarks {}'.format(self.median_landmarks))
			median_landmarks_vals = []
			for mlm in self.median_landmarks:
				self.selected_algo[mlm] = self.M_test[mlm]
				self.to_select_algo.remove(mlm)
				median_landmarks_vals.append(self.selected_algo[mlm])
				# max_landmark_val = np.max(landmarks_vals)
				# self.best_so_far = [max_landmark_val]*self.num_landmarks
			for mlv_idx in range(len(median_landmarks_vals)-1):
				if median_landmarks_vals[mlv_idx+1] < median_landmarks_vals[mlv_idx]:
					median_landmarks_vals[mlv_idx+1] = median_landmarks_vals[mlv_idx]

			self.best_so_far = median_landmarks_vals
			self.next_algo = self.median_rank[len(self.selected_algo)] # s

			self.best_so_far = median_landmarks_vals

			module_logger.info('Setup Cofirank initialized by Median')
			self.num_cofi_run += 1
			M_train_median_Cofi = np.vstack((self.M_train, self.M_test))
			for c in range(len(self.M_train[-1,:])):
				if c in self.median_landmarks:
					M_train_median_Cofi[-1, c] = self.M_test[c]
				else:
					M_train_median_Cofi[-1, c] = np.nan
				# print 'After putting median landmarks, M_train_median_Cofi last row looks like: ', M_train_median_Cofi[-1, :]
			matrix_to_lsvm(M_train_median_Cofi, row_range=(range(M_train_median_Cofi.shape[0])), start_idx=0, \
					save_name=self.M_name+'_train.lsvm', save_dir=self.Cofi['medianData_dir'])
			M_test_median_Cofi = self.M_test

			matrix_to_lsvm(M_test_median_Cofi, row_range=(range(M_train_median_Cofi.shape[0])), start_idx=M_train_median_Cofi.shape[0]-1, \
				save_name=self.M_name+'_test.lsvm', save_dir=self.Cofi['medianData_dir'])
			# os.chdir(CofiRank_dir)
			os.chdir(self.Cofi['dir'])
			make_config_file(config_dir=self.Cofi['config_dir'], new_config_filename=self.Cofi['config_file_median'], \
				DtrainFile=os.path.join(self.Cofi['medianData_reldir'], self.M_name+'_train.lsvm'), \
				DtestFile=os.path.join(self.Cofi['medianData_reldir'], self.M_name+'_test.lsvm'), \
				outfolder=self.Cofi['medianOut_reldir'])
			call(['./dist/cofirank-deploy', 'config/%s'%self.Cofi['config_file_median']])
			F_median = lsvm_to_matrix(M_train_median_Cofi.shape, 'F.lsvm', lsvm_dir=self.Cofi['medianOut_dir'])
			F_median_test = F_median[-1, :]
			self.rank_median_test = list(reversed(np.argsort(F_median_test)))
			module_logger.debug('Cofi suggests ranking {}'.format(self.rank_median_test))
			self.current_rank = self.rank_median_test

			for top_alg in self.rank_median_test:
				module_logger.debug('considering {}'.format(top_alg))
				if not top_alg in self.selected_algo.keys():
					module_logger.debug('will choose as self.next_algo'.format(top_alg))
					self.next_algo = top_alg
					break
			print 'I am ok????'
			print self.next_algo, self.num_cofi_run, self.current_rank, self.median_landmarks
			
			return self.next_algo, self.num_cofi_run, self.current_rank, self.median_landmarks

	def	svd_landmarks_with_1_cofirank(self):
		pass



