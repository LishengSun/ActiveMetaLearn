import numpy as np
import os
import matplotlib.pyplot as plt

def make_SVD_artificial_matrix(num_row, num_col, rank):
	"""
	Make a matrix with shape (num_row, num_col) and rank = rank
	M = U S V
	where S is diagonal with values decreasing exponentially
	"""
	if rank > min(num_row, num_col):
		raise ValueError('Rank not realizable.')

	from scipy.stats import ortho_group
	U = ortho_group.rvs(dim=num_row)
	# print np.dot(U[:,0],U[:,0].T)
	# print np.dot(U[:,1],U[:,0].T)
	V = ortho_group.rvs(dim=num_col)

	S = np.zeros((num_row, num_col))
	for d in range(rank): # make exponential decay
		S[d,d] = 100*np.exp(-d)

	M = np.dot(U, np.dot(S, V))
	return S, M



if __name__ == '__main__':
	S, M = make_SVD_artificial_matrix(50,20,20)
	S_diago = np.diag(S)
	# x = range(len(S_diago))
	# plt.bar(x, S_diago, label='Diag(d) = 100 * exp(-d)')
	# plt.xticks(x, range(1,len(S_diago)+1))
	# plt.ylabel('Diagonal values of S')
	# plt.xlabel('d')
	# plt.title('artificial matrix (50*20) M = USV')
	# plt.legend()
	# plt.savefig(os.path.join(os.getcwd(),'./artificial/r50c20r20_S_diagonal'))
	# plt.show()
	# np.savetxt(os.path.join(os.getcwd(),'./artificial/r50c20r20.data'), M)








