import h5py
import numpy as np
from .Sim_info import SimInfo
from .write_data import *


class ReadData(SimInfo):
	"""
	Class to read different hdf5 data files.
	Assumes underlying file structures used in MeasureIA and MeasureSnapshotVariables classes.

	Attributes
	----------
	catalogue : str
		Catalogue name that contains the data.
	sub_group : str, optional
			Name of group(s)/structure within snap_group where dataset is found. Default is empty str.
	output_file_name : str, optional
			Name where output should be stored.
	data_path : str, optional
		The path to where the data is saved. Default='./data/raw/

	Notes
	-----
	Inherits attributes from 'SimInfo', where 'snap_group', 'snap_folder' and 'fof_folder' are used in this class.

	"""

	def __init__(
			self, simulation, catalogue, snapshot, sub_group="", output_file_name=None, data_path="./data/raw/"
	):
		"""
		The __init__ method of the ReadData class.

		Parameters
		----------
		simulation : str
			Identifier of the simulation, allowing for correct information to be obtained.
			Choose from [TNG100, TNG100_2, TNG300, EAGLE, HorizonAGN, FLAMINGO_L1, FLAMINGO_L2p8].
		catalogue : str
			Catalogue name that contains the data. If groupcat file: 'Subhalo' (then use read_subhalo).
			If snapshot file: enter 'PartTypeX' where X is the particle type number (then use read_snapshot).
		snapshot : int or str or NoneType
			Number of the snapshot.
		sub_group : str, optional
			Name of group(s)/structure within snap_group where dataset is found. Default is empty str.
		output_file_name : str, optional
			Name where output should be stored.
		data_path : str, optional
			The path to where the data is saved. Default='./data/raw/

		"""
		SimInfo.__init__(self, simulation, snapshot, boxsize=None, file_info=True)
		self.catalogue = catalogue
		self.sub_group = sub_group + "/"
		self.data_path = data_path + "/"
		self.output_file_name = output_file_name
		return

	def read_cat(self, dataset_name, cut=None):
		"""Reads the data from the specified catalogue.

		Parameters
		----------
		dataset_name :
			the dataset name for the requested data
		cut : iterable with 2 entries
			 Read dataset slice [cut[0]:cut[1]]. Default value = None

		Returns
		-------
		ndarray
			The requested dataset (slice)

		Raises
		------
		KeyError
			If catalogue=Subhalo or Snapshot.

		"""
		if self.catalogue == "Subhalo":
			raise KeyError("Use read_subhalo method")
		elif self.catalogue == "Snapshot":
			raise KeyError("Use read_snapshot method")

		file = h5py.File(f"{self.data_path}{self.catalogue}.hdf5", "r")
		if cut == None:
			data = file[self.snap_group + self.sub_group + dataset_name][:]
		else:
			data = file[self.snap_group + self.sub_group + dataset_name][cut[0]: cut[1]]
		return data

	def read_subhalo(self, dataset_name, Nfiles=0):
		"""Read the data from the subhalo files.

		Parameters
		----------
		dataset_name :
			The dataset name for the requested data
		Nfiles : int, optional
			 Number of files to read from. Default=0, in which case the number from SimInfo object is used.

		Returns
		-------
		ndarray
			The requested dataset

		"""
		subhalo_file = h5py.File(f"{self.data_path}{self.fof_folder}.0.hdf5", "r")
		Subhalo = subhalo_file[self.catalogue]
		try:
			data = Subhalo[dataset_name][:]
		except KeyError:
			print("Variable not found in Subhalo files. Choose from ", Subhalo.keys())
		if len(np.shape(data)) > 1:
			stack = True
		else:
			stack = False
		subhalo_file.close()
		if Nfiles == 0:
			Nfiles = self.N_files

		for n in np.arange(1, Nfiles):
			subhalo_file = h5py.File(f"{self.data_path}{self.fof_folder}.{n}.hdf5", "r")
			try:
				Subhalo = subhalo_file[self.catalogue]
				data_n = Subhalo[dataset_name][:]  # get data single file
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

	def read_snapshot(self, dataset_name):
		"""Read the data from the snapshot files and optionally write to output file.

		Parameters
		----------
		dataset_name :
			The dataset name for the requested data

		Returns
		-------
		ndarray
			The requested dataset or nothing if output_file_name is specified

		"""
		if self.output_file_name != None:
			output_file = h5py.File(self.output_file_name, "a")
			group_out = create_group_hdf5(output_file, self.snap_group)
			write_output = True
		else:
			write_output = False
		print(dataset_name)
		snap_file = h5py.File(f"{self.data_path}{self.snap_folder}.0.hdf5", "r")
		Snap_data = snap_file[self.catalogue]

		try:
			data = Snap_data[dataset_name][:]
		except KeyError:
			print(f"Variable not found in Snapshot files: {dataset_name}. Choose from ", Snap_data.keys())
		if len(np.shape(data)) > 1:
			stack = True
		else:
			stack = False
		if write_output:
			try:
				dataset = group_out[dataset_name]
				del group_out[dataset_name]
			except:
				pass
			if stack:
				group_out.create_dataset(dataset_name, data=data, maxshape=(None, np.shape(data)[1]), chunks=True)
			else:
				group_out.create_dataset(dataset_name, data=data, maxshape=(None,), chunks=True)
		snap_file.close()

		for n in np.arange(1, self.N_files):
			snap_file = h5py.File(f"{self.data_path}{self.snap_folder}.{n}.hdf5", "r")
			try:
				Snap_data = snap_file[self.catalogue]
				data_n = Snap_data[dataset_name][:]  # get data single file
			except KeyError:
				print("problem at file ", n)
				snap_file.close()
				continue
			if write_output:
				group_out[dataset_name].resize((group_out[dataset_name].shape[0] + data_n.shape[0]), axis=0)
				group_out[dataset_name][-data_n.shape[0]:] = data_n
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

	def read_snapshot_multiple(self, dataset_name):
		"""Read multiple datasets from the snapshot files for a specified shapshot number.

		Parameters
		----------
		dataset_name : list or str
			The dataset names for the requested data

		Returns
		-------
		type
			The requested datasets or nothing if output_file_name is specified

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
		for i, variable in enumerate(dataset_name):
			try:
				data = Snap_data[dataset_name[i]][:]
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
			for i, variable in enumerate(dataset_name):
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


if __name__ == "__main__":
	pass
