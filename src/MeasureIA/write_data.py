def write_dataset_hdf5(group, name, data):
	"""Creates the desired dataset, overwriting if there is already one with that name.

	Parameters
	----------
	group : <class 'h5py._hl.group.Group'>
		link to group in datafile
	name :
		name of desired dataset
	data :
		the data

	"""
	try:
		dataset = group[name]
		del group[name]
		group.create_dataset(name, data=data)
	except:
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
	try:
		group_file = file[name]
		groups = name
	except:
		list_groups = name.split('/')
		groups = ''
		for group in list_groups:
			if groups == '':
				groups += group
			else:
				groups += '/'
				groups += group
			try:
				group_file = file[groups]
			except:
				file.create_group(groups)
	if groups != name:
		raise ValueError('Created groups do not match specified one.')
	return file[name]


if __name__ == "__main__":
	pass
