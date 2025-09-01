def write_dataset_hdf5(group, name, data):
	"""Deletes a dataset with the name of the dataset we want to create if it exists before creating the desired dataset.

	Parameters
	----------
	group :
		link to group in datafile
	name :
		name of desired dataset
	data :
		the data

	Returns
	-------

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
	file :
		hdf5 file
	name :
		path of groups (e.g. group1/group2)

	Returns
	-------
	type
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