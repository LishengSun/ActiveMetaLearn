import os





def make_config_file(config_dir, new_config_filename, DtrainFile, DtestFile, outfolder, old_config_filename='my_default.cfg'):
	"""
	make new config file based on an existing one, default = default.cfg
	replace DtrainFile (line 7), DtestFile (line 8), outfolder (line 12)
	"""
	f_old = open(os.path.join(config_dir, old_config_filename), 'r')
	f_new = open(os.path.join(config_dir, new_config_filename), 'w')
	for i, line in enumerate(f_old, 1):
		if i in range(1,7)+range(9,12)+range(13, 72):
			f_new.write(line)
		elif i == 7:
			f_new.write('string cofibmrm.DtrainFile %s \n'%DtrainFile)
			# f_new.write('string cofibmrm.DtrainFile data/dummytrain.lsvm \n')
		elif i == 8:
			f_new.write('string cofibmrm.DtestFile %s \n'%DtestFile)
		elif i == 12:
			f_new.write('string cofi.outfolder %s \n'%outfolder)
	f_old.close()
	f_new.close()