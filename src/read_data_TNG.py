import h5py
import numpy as np
from src.Sim_info import SimInfo
from src.write_data import *


class ReadData(SimInfo):
	"""
	Class to read different TNG data files.
	:param project (str): Project name. Choose 'TNG100' or 'TNG300'
	:param catalogue (str): Catalogue name that contains the data. If groupcat file: 'Subhalo'.
													If snapshot file: enter 'PartTypeX' where X is the particle type number.
	:param output_file_name: file path and name where output should be stored
	:param snapshot (int): The number of the snapshot. e.g. 99
	:param data_path (str): The start path to where the data is saved. Expects the files to be in a folder with the project name.
	"""

	def __init__(
			self, project, catalogue, snapshot, sub_group="", output_file_name=None, data_path="./data/raw/",
			sub_folder="", manual=False, update=False
	):
		SimInfo.__init__(self, project, snapshot, manual=manual, update=update)
		self.catalogue = catalogue
		self.sub_group = sub_group
		self.data_path = data_path + project + "/" + sub_folder + "/"
		self.output_file_name = output_file_name
		return

	def read_cat(self, variable, cut=None):
		"""
		Reads the data from the specified catalogue for a specified snapshot.
		:param variable: the variable name for the requested data
		:return: the data
		"""
		if self.catalogue == "Subhalo":
			raise KeyError("Use ReadSubhalo method")
		elif self.catalogue == "Snapshot":
			raise KeyError("Use ReadSnapshot method")

		file = h5py.File(f"{self.data_path}{self.catalogue}.hdf5", "r")
		if cut == None:
			data = file[self.snap_group + self.sub_group + variable][:]
		else:
			data = file[self.snap_group + self.sub_group + variable][cut[0]: cut[1]]
		return data

	def read_subhalo(self, variable):
		"""
		Read the data from the subhalo files for a specified shapshot.
		:param variable: the variable name for the requested data
		:return: the data
		"""
		subhalo_file = h5py.File(f"{self.data_path}{self.fof_folder}.0.hdf5", "r")
		Subhalo = subhalo_file[self.catalogue]
		try:
			data = Subhalo[variable][:]
		except KeyError:
			print("Variable not found in Subhalo files. Choose from ", Subhalo.keys())
		if len(np.shape(data)) > 1:
			stack = True
		else:
			stack = False
		subhalo_file.close()

		for n in np.arange(1, self.N_files):
			subhalo_file = h5py.File(f"{self.data_path}{self.fof_folder}.{n}.hdf5", "r")
			try:
				Subhalo = subhalo_file[self.catalogue]
				data_n = Subhalo[variable][:]  # get data single file
			except KeyError:
				print("problem at file ", n)
				subhalo_file.close()
				continue
			if stack:
				data = np.vstack((data, data_n))
			else:
				data = np.append(data, data_n)
			subhalo_file.close()
		return data

	def read_snapshot(self, variable):
		"""
		Read the data from the snapshot files for a specified shapshot number
		:param variable: the variable name for the requested data
		:return: the data or nothing if output_file_name is specified
		"""
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group_out = create_group_hdf5(output_file, self.snap_group)
			write_output = True
		else:
			write_output = False
		print(variable)
		snap_file = h5py.File(f"{self.data_path}{self.snap_folder}.0.hdf5", "r")
		Snap_data = snap_file[self.catalogue]

		try:
			data = Snap_data[variable][:]
		except KeyError:
			print(f"Variable not found in Snapshot files: {variable}. Choose from ", Snap_data.keys())
		if len(np.shape(data)) > 1:
			stack = True
		else:
			stack = False
		if write_output:
			try:
				dataset = group_out[variable]
				del group_out[variable]
			except:
				pass
			if stack:
				group_out.create_dataset(variable, data=data, maxshape=(None, np.shape(data)[1]), chunks=True)
			else:
				group_out.create_dataset(variable, data=data, maxshape=(None,), chunks=True)
		snap_file.close()

		for n in np.arange(1, self.N_files):
			snap_file = h5py.File(f"{self.data_path}{self.snap_folder}.{n}.hdf5", "r")
			try:
				Snap_data = snap_file[self.catalogue]
				data_n = Snap_data[variable][:]  # get data single file
			except KeyError:
				print("problem at file ", n)
				snap_file.close()
				continue
			if write_output:
				group_out[variable].resize((group_out[variable].shape[0] + data_n.shape[0]), axis=0)
				group_out[variable][-data_n.shape[0]:] = data_n
			else:
				if stack:
					data = np.vstack((data, data_n))
				else:
					data = np.append(data, data_n)
			snap_file.close()
		if write_output:
			output_file.close()
			return
		else:
			return data

	def read_snapshot_multiple(self, variables):
		"""
		Read the data from the snapshot files for a specified shapshot number
		:param variable: the variable name for the requested data
		:return: the data or nothing if output_file_name is specified
		"""
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group_out = create_group_hdf5(output_file, self.snap_group)
			write_output = True
		else:
			write_output = False
		snap_file = h5py.File(f"{self.data_path}{self.snap_folder}.0.hdf5", "r")
		Snap_data = snap_file[self.catalogue]
		stack = []
		for i, variable in enumerate(variables):
			try:
				data = Snap_data[variables[i]][:]
			except KeyError:
				print(f"Variable not found in Snapshot files {variable}. Choose from ", Snap_data.keys())
			if len(np.shape(data)) > 1:
				stack.append(True)
			else:
				stack.append(False)
			if write_output:
				try:
					dataset = group_out[variable]
					del group_out[variable]
				except:
					pass
				if stack[i]:
					group_out.create_dataset(variable, data=data, maxshape=(None, np.shape(data)[1]), chunks=True)
				else:
					group_out.create_dataset(variable, data=data, maxshape=(None,), chunks=True)

		snap_file.close()

		for n in np.arange(1, self.N_files):
			snap_file = h5py.File(f"{self.data_path}{self.snap_folder}.{n}.hdf5", "r")
			for i, variable in enumerate(variables):
				try:
					Snap_data = snap_file[self.catalogue]
					data_n = Snap_data[variable][:]  # get data single file
				except KeyError:
					print("problem at file ", n)
					snap_file.close()
					continue
				if write_output:
					group_out[variable].resize((group_out[variable].shape[0] + data_n.shape[0]), axis=0)
					group_out[variable][-data_n.shape[0]:] = data_n
				else:
					if stack[i]:
						data = np.vstack((data, data_n))
					else:
						data = np.append(data, data_n)
			snap_file.close()
		if write_output:
			output_file.close()
			return
		else:
			return data
