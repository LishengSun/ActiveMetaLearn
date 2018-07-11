from numpy import random
import os
import sys
from os import path
import numpy as np
import random






def matrix_to_lsvm(M, row_range, start_idx, save_name, save_dir='/Users/lishengsun/Dropbox/Meta-RL/cofirank/data/lisheng-data/'):
	"""
	Turn a matrix (with nan) to lsvm format to feed CofiRank
	for example: [0.1, 0.4, nan, 0.9] will become 1:0.1 2:0.4 4:0.9
	Attention: must start from 1 (requirement from CofiRank)
	"""
	with open(os.path.join(save_dir, save_name), 'w') as f: 
		for r in row_range:
			if r < start_idx:
				f.write('\n')
			else:
				if len(M.shape) == 1: 
					for c in range(len(M)):
						if not np.isnan(M[c]):
							f.write(str(c+1)+':'+str(M[c])+'\t')
					f.write('\n')
				else: 
					for c in range(M.shape[1]):
						if not np.isnan(M[r, c]):
						# print M[r, c]
							f.write(str(c+1)+':'+str(M[r, c])+'\t')
					f.write('\n')

		# print os.path.join(save_dir, save_name)
		# if len(M.shape) == 1: # case of 1 test 
		# 	for i in range(len(M)):
		# 		if not np.isnan(M[i]):
		# 			f.write(str(i+1)+':'+str(M[i])+'\t')
		# 	# print "Do not give me matrix like (n,)! "
		# else:
		# 	for r in range(M.shape[0]):
		# 		for c in range(M.shape[1]):
		# 			if not np.isnan(M[r, c]):
		# 				# print M[r, c]
		# 				f.write(str(c+1)+':'+str(M[r, c])+'\t')
		# 		f.write('\n')

def lsvm_to_matrix(matrix_shape, lsvm_file, lsvm_dir='/Users/lishengsun/Dropbox/Meta-RL/cofirank/out_lisheng'):
	"""
	Turn lsvm to array
	Recover U, M from CofiRank output
	"""
	M = []
	with open(os.path.join(lsvm_dir, lsvm_file), 'r') as f:
		for line in f:
			x = [float(item.split(':')[-1]) for item in line.split()]
			if x == []: M.append([np.nan]*matrix_shape[1])
			else: M.append(x)
			# print x
	M = np.asarray(M)

	return M

def _sparsity(M):
	"""
	compute frac_miss of matrix M
	"""
	sparsity = float(np.count_nonzero(np.isnan(X))) / np.size(X)
	return sparsity

def _missing_positions(M):
	"""
	return positions of missing values in matrix M
	"""
	sparsity = _sparsity(M)
	if sparsity == 0:
		missing_positions = None
	else:
		missing_positions = np.argwhere(np.isnan(x))
	return missing_positions

def split_matrix_to_cofiTrainTest(M, sparsify_M=True, frac_miss=0.5):
	"""
	if M has missing values => M_train_CofiRank=known_values, M_test_CofiRank=missing_values
	"""
	M_train_CofiRank = np.copy(M)
	M_test_CofiRank = np.copy(M)
	M_list = M.reshape(-1,).tolist()
	M_train_list = M_train_CofiRank.reshape(-1,).tolist()
	M_test_list = M_test_CofiRank.reshape(-1,).tolist()
	# print M_train_CofiRank.shape, len(M_train_list)
	# print M_test_CofiRank.shape, len(M_test_list)
	if sparsify_M == True:
		num_miss = int(frac_miss * M.size)	
		nan_idx = random.sample(range(len(M_list)), num_miss)
	else:
		nan_idx = [i for i in range(len(M_list)) if M_list[i] == np.nan]
	for ii in nan_idx:
		M_train_list[ii] = np.nan
	for i in range(len(M_test_list)):
		if not i in nan_idx:
			M_test_list[i] = np.nan 
	# print len(M_train_list), len(M_test_list)
	M_train_CofiRank = np.reshape(M_train_list, M_train_CofiRank.shape)
	M_test_CofiRank = np.reshape(M_test_list, M_test_CofiRank.shape)

	return M_train_CofiRank, M_test_CofiRank