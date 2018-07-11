import numpy as np


def standardize(X):
	"""
	column wise normalization, based on Isabelle's Matlab code
	"""
	XX = np.copy(X) # no destroy X
	for iteration in range(5):
		mu_row = np.nanmean(XX, axis=1) # mean of row
		std_row = np.nanstd(XX, axis=1)
		# print 'mu_row = ', mu_row
		# print 'std_row = ', std_row
		for r in range(XX.shape[0]):
			# if sparsity(mu_row)==0 and sparsity(std_row)==0:

			XX[r,:] = (XX[r,:]-mu_row[r])/(0.1*std_row[r])
			# else:
			# 	pass
		mu_col = np.nanmean(XX, axis=0) # mean of col
		std_col = np.nanstd(XX, axis=0)
		for c in range(XX.shape[1]):
			# if sparsity(mu_col) == 0 and sparsity(std_col)==0:
			XX[:,c] = (XX[:,c]-mu_col[c])/(0.1*std_col[c])
			# else:
			# 	pass
	return XX