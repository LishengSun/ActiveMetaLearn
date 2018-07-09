import os
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from os import path
# sys.path.append(path.abspath('../utils/'))
from standardize_matrix import standardize
from matrix_io import *
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable




class Displayer(object):
	"""
	Visualization tools for a performance matrix.
	Inputs: 
			- M: array. The input performance matrix
			- name: string. Will be used as title for plots
			- save: bool. Whether to save the plots. If true, use save_dir to indicate the path.
			- global_norm: bool. Wheter to normalize the matrix globally before any visualization. True by default.
			- log: bool. Wheter to take natural logarithm of the matrix before any visualization. False by default.
			- metric_error: bool. By default we suppose the performance (i.e. values of M) represents some `scores' 
			for which higher is better. If metric_error set to True, then lower is better and -M (rather than M)
			will be displyed.
	"""
	def __init__(self, M, name, save, save_dir=None, global_norm=True, log=False, metric_error=False):
		super(Displayer, self).__init__()
		self.M = M
		self.name = name
		self.save = save
		self.global_norm = global_norm
		self.log = log
		self.metric_error = metric_error

		if self.log:
			self.M[np.where(self.M==0)] += 0.0001
			self.M = np.log(self.M)
			self.name += '_log'
		if self.global_norm:
			self.M = (self.M - np.mean(self.M)) / np.std(self.M)
			self.name += '_gn'
		if self.metric_error:
			self.M = -self.M

		if self.save == True:
			self.save_dir = save_dir

 		
 		self.vmin = -max(np.abs(np.min(self.M)), np.abs(np.max(self.M)))
 		self.vmax = max(np.abs(np.min(self.M)), np.abs(np.max(self.M)))


 		
		# min_bound = np.nanmin(self.M)
		# max_bound = np.nanmax(self.M)

		
 	def plot_raw_matrix_with_missing_values(self):
 		"""
 			Display the raw matrix with missing values shown as blank
 		"""
 		fig, ax = plt.subplots()
 		cax = ax.imshow(self.M, origin='lower', cmap="Blues", vmin=self.vmin, vmax=self.vmax)
 		fig.colorbar(cax)
 		plt.xlabel('algorithms')
		plt.ylabel('datasets')
		plt.title('%s : raw matrix'%self.name)
		if self.save:
			plt.savefig(os.path.join(self.save_dir,'raw_matrix_%s_median'%self.name))
		plt.show()



	def plot_matrix_with_missing_value_median(self):
		"""
			Display the matrix with columns arranged in descending median.
		"""
		
		fig, ax = plt.subplots()

		self.M_median = np.copy(self.M)
		self.median_rank = list(reversed(np.argsort(np.nanmedian(self.M, axis=0))))
		# print np.nanmedian(self.M, axis=0)
		# print self.median_rank
		for col in range(self.M.shape[1]):
			self.M_median[:, col] = self.M[:, self.median_rank[col]]
		
		cax = ax.imshow(self.M_median, origin='lower', cmap="Blues", vmin=self.vmin, vmax=self.vmax)#, cmap=cmap, norm=norm)

		fig.colorbar(cax)
		for row in range(self.M_median.shape[0]):
			rmax = np.max(self.M_median[row,:])
			row_max = np.where(self.M_median[row,:]==rmax)[0]
			# print (row_max, row)
			for rm in row_max:
				plt.plot(rm, row, 'o', markersize=2, color='red')

		# print self.M_median[:,0]-self.M[:,self.median_rank[0]]
		ax.set_xticks(range(self.M.shape[1]))#len(datanames)+1))
		ax.tick_params(axis='both', which='both',length=0)
		ax.set_xticklabels(self.median_rank, fontsize=5)
		plt.xlabel('algorithms')
		plt.ylabel('datasets')
		plt.title('%s : MEDIAN-RANKED COL'%self.name)
		

		if self.save:
			plt.savefig(os.path.join(self.save_dir,'raw_matrix_%s_median'%self.name))
		plt.show()


	def plot_matrix_with_missing_value_svd(self):
		"""
			Display the matrix with columns arranged in descending svd projection.
		"""
		
		self.svd_rank = []
		U, s, V = np.linalg.svd(self.M)
		pcomp = U[:self.M.shape[1],:]
		for pc_idx in range(len(pcomp)):
			col_weight_pc = []
			for M_col in range(self.M.shape[1]): # for each column of M_train, test its projection to U's columns
				cw = abs(np.inner(self.M[:,M_col], pcomp[pc_idx]))
				col_weight_pc.append(cw)
			for c in sorted(range(len(col_weight_pc)), key=lambda k: col_weight_pc[k], reverse=True):
				if not c in self.svd_rank:
					self.svd_rank.append(c)
					break
			
		fig, ax = plt.subplots()

		self.M_svd = np.copy(self.M)

		for col in range(len(self.svd_rank)):
			self.M_svd[:, col] = self.M[:, self.svd_rank[col]]
		if self.M.shape[1] > len(self.svd_rank):
			last_i=0
			for c in range(self.M.shape[1]): 
				if not c in self.svd_rank:
					self.M_svd[:, len(self.svd_rank)+last_i] = self.M[:,c] 
					last_i += 1
		
		cax = ax.imshow(self.M_svd, origin='lower', cmap="Blues", vmin=self.vmin, vmax=self.vmax)

		fig.colorbar(cax)
		for row in range(self.M_svd.shape[0]):
			rmax = np.max(self.M_svd[row,:])
			row_max = np.where(self.M_svd[row,:]==rmax)[0]
			for rm in row_max:
				plt.plot(rm, row, 'o', markersize=2, color='red')
		ax.set_xticks(range(self.M.shape[1]))#len(datanames)+1))
		ax.tick_params(axis='both', which='both',length=0)
		ax.set_xticklabels(self.svd_rank, fontsize=5)
		plt.xlabel('algorithms')
		plt.ylabel('datasets')
		plt.title('%s : SVD-RANKED COL'%self.name)
		

		if self.save:
			plt.savefig(os.path.join(self.save_dir,'raw_matrix_%s_svd'%self.name))
		plt.show()	

	
	def plot_spectrum(self):
		"""
			Display singular values in descending order.
		"""
		U,s,V = np.linalg.svd(self.M)
		x = range(len(s))
		if 'artificial' in self.name:
			plt.bar(x, s, label='Diag(d) = 100 * exp(-d)')
			# plt.bar(x, s)
		else:
			plt.bar(x, s)
		plt.plot(x, s, 'o-', color='red')
		plt.xticks(x, range(1,len(s)+1), fontsize=2)
		plt.ylabel('singular values')
		plt.xlabel('d')
		plt.title(self.name)
		# plt.title('OpenML-Alors')
		plt.legend()
		if self.save:
			plt.savefig(os.path.join(self.save_dir,'spectrum_%s'%self.name))
		plt.show()




	def plot_hierarchical_clustering(self, metric='euclidean'):
		"""
			Clustering calculated based on metric, default = euclidean.
			For more metric options see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
		"""
		import seaborn as sns
		sns.set(color_codes=True)
		cg = sns.clustermap(self.M, metric=metric, cmap="Blues")
		# cg.dendrogram_col.linkage
		# cg.dendrogram_row.linkage
		plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
		plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=2)

		plt.xlabel('algorithms')
		plt.ylabel('datasets')
		plt.title(self.name)
		# plt.title('OpenML-Alors')

		if self.save:
			plt.savefig(os.path.join(self.save_dir,'hierarchical_clustering_%s'%self.name))
		plt.show()



	def display_all(self):
		self.plot_raw_matrix_with_missing_values()
		self.plot_matrix_with_missing_value_svd()
		self.plot_matrix_with_missing_value_median()
		self.plot_hierarchical_clustering()
		self.plot_spectrum()




if __name__ == '__main__':
	data_dir = '/Users/lishengsun/Dropbox/Meta-RL/APT-PAPER_PKDD2018/DATASETS/'

	# M = np.loadtxt(os.path.join(data_dir, 'artificial/r50c20r20.data'))
	# name = 'artificial'
	fig_dir = '/Users/lishengsun/Dropbox/Meta-RL/APT-PAPER_PKDD2018/PAPER-FIG/Displayer'
	M = np.loadtxt(os.path.join(data_dir, 'OpenML-Alors/openml-ai-accuracy.data'))
	name = 'ALLOpenML-Alors'
	# 

	# M = np.loadtxt(os.path.join(data_dir, 'AutoML_sameLoss/AutoML.data'))
	# M = M[:,9:]
	# name = 'AutoML'
	# M = np.loadtxt(os.path.join(data_dir, 'statlog/statlogwithds18.data'))
	# name = 'Statlog'
	# M = np.sqrt(M)
	# M = M[:60, :5]
	# np.savetxt(os.path.join(data_dir, 'OpenML/OpenML_60-20_FULL.data'), M)
	# M = np.loadtxt(os.path.join(data_dir, 'OpenML/OpenML_60-20_FULL.data'))
	# name = 'OpenML'
	# M = standardize(M)
	# np.savetxt(os.path.join(data_dir, 'statlog/statlog_stand_std10.data'), M)
	D = Displayer(M, name, save=True, global_norm=True, log=False, metric_error=True)
	D.display_all()





