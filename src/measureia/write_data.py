def write_dataset_hdf5(group, name, data):
	"""Creates the desired dataset, overwriting if there is already one with that name.

	Parameters
	----------
	group : <class 'h5py._hl.group.Group'>
		link to group in datafile
	name : str
		name of desired dataset
	data : ndarray
		the data

	"""
	if name in group:
		del group[name]
	group.create_dataset(name, data=data)
	return


def create_group_hdf5(file, name):
	"""Checks if path of groups within hdf5 file exists. If not, builds the groups one by one.

	Parameters
	----------
	file : <class 'h5py._hl.files.File'>
		hdf5 file
	name : str
		path of groups (e.g. group1/group2)

	Returns
	-------
	<class 'h5py._hl.group.Group'>
		link to the last group in the path

	"""
	if name not in file:
		list_groups = name.split('/')
		groups = ''
		for group in list_groups:
			if group != '':
				if groups == '':
					groups += group
				else:
					groups += '/'
					groups += group
				if groups not in file:
					file.create_group(groups)
	return file[name]


if __name__ == "__main__":
	pass
