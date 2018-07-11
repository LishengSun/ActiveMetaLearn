import random
import matplotlib.pyplot as plt 
import numpy as np
import os
import sys
from os import path


import json
import seaborn as sns; sns.set(color_codes=True)



def plot_multi_bands(X, axis=0, x_starts_from_1=True, color='blue', label='Random'):
	"""
	Given a matrix X of shape (m, n)
	We plot median, quartile, min, max (over axis) with envelops of decreasing shading colors
	"""
	fig, ax = plt.subplots()
	# palette = sns.color_palette()
	lower = np.min(X, axis=0)
	upper = np.max(X, axis=0)
	median = np.median(X, axis=0)
	Q5 = np.percentile(X, 5, axis=0)
	Q25 = np.percentile(X, 25, axis=0)
	Q75 = np.percentile(X, 75, axis=0)
	Q95 = np.percentile(X, 95, axis=0)
	if x_starts_from_1:
		x_axis = range(1,X.shape[1]+1)
	else:
		x_axis = range(X.shape[1])
	ax.plot(x_axis, median, 's-', color='blue', markersize=4, label=label+': median')
	ax.plot(x_axis, lower, alpha=.1, color=color, label=label+': min')
	# ax.plot(x_axis, upper, alpha=.1, color=color)
	ax.fill_between(x_axis, Q5, Q25, alpha=.15, color=color, label=label+': 5%')
	ax.fill_between(x_axis, Q25, median, alpha=.2, color=color, label=label+': 25%')
	ax.fill_between(x_axis, median, Q75, alpha=.25, color=color, label=label+': 75%')
	ax.fill_between(x_axis, Q75, Q95, alpha=.3, color=color, label=label+': 95%')
	# ax.fill_between(x_axis, median-Q1, median+Q1, alpha=.2, color=color)
	# ax.fill_between(x_axis, median-Q3, median+Q3, alpha=.15, color=color)
	# ax.fill_between(x_axis, lower, upper, alpha=.2, color='yellow')
	return fig, ax





if __name__ == '__main__':
	import json

	R = json.load(open('../intermediate-experiments/artificial_r50c20r20Random_best_so_far_alltest.txt'))
	R = R['test_0']
	R = np.array(R).reshape(1000, -1)
	fig, ax = plot_multi_bands(R)
	fig.show()






