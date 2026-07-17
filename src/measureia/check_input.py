import os
import warnings
import numpy as np


class CheckInput:
	"""Class that contains static methods to validate the input data of MeasureIABox and MeasureIALightcone and to
	translate user-defined data dictionary key names to the internal (default) names.

	Methods
	-------
	check_dict()
		Check that a data dictionary contains all required keys.
	check_paths()
		Check that the folder of each given file path exists.
	check_units_coordinates()
		Check that coordinates fall within [0, boxsize].
	check_type_input_data()
		Check types and shapes of the entries of the input data dictionary.
	check_jackknife_max_separation()
		Warn if the maximum separation or number of r bins is too large for the jackknife subbox size.
	rename_input_keys()
		Return a shallow copy of a data dictionary with user-defined key names replaced by the internal names.

	"""

	@staticmethod
	def check_dict(dict, names):
		"""Checks that all keys in 'names' exist in the given dictionary.

		Parameters
		----------
		dict : dict
			Data dictionary to check.
		names : iterable of str
			Names of the keys that must be present.

		"""
		for name in names:
			if name not in dict:
				raise KeyError(f"Your data dictionary does not contain {name}.")
		return

	@staticmethod
	def check_paths(paths):
		"""Checks that the folder of each given file path exists.

		Parameters
		----------
		paths : iterable of str
			File paths whose parent folders are checked.

		"""
		for path in paths:
			folder_path = os.path.dirname(path) or "."
			if not os.path.exists(folder_path):
				raise FileNotFoundError(f"{path} does not exist.")
		return

	@staticmethod
	def check_units_coordinates(coordinates, boxsize):
		"""Checks that the coordinates fall within [0, boxsize].

		Parameters
		----------
		coordinates : ndarray
			(N,3) array of positions.
		boxsize : float
			Size of the simulation box, in the same units as the coordinates.

		"""
		for i in np.arange(0, len(coordinates[0])):
			if min(coordinates[:, i]) < 0. or max(coordinates[:, i]) > boxsize:
				raise ValueError(
					"The coordinates do not agree with the boxsize. They should be in range [0, boxsize]. Check the units.")
		return

	@staticmethod
	def check_type_input_data(dict, names):
		"""Checks types and shapes of the entries of the input data dictionary for MeasureIABox.

		Parameters
		----------
		dict : dict
			Data dictionary to check.
		names : iterable of 5 str
			Key names of the positions of the density sample, positions of the shape sample, axis directions,
			axis ratios and line-of-sight index, in that order.

		"""
		(positions_density_sample_name, positions_shape_sample_name, axis_direction_name, axis_ratio_name,
		 line_of_sight_index_name) = names
		assert type(dict[positions_density_sample_name]) == np.ndarray
		assert np.shape(dict[positions_density_sample_name])[1] == 3
		assert type(dict[positions_shape_sample_name]) == np.ndarray
		assert np.shape(dict[positions_shape_sample_name])[1] == 3
		assert type(dict[axis_direction_name]) == np.ndarray
		assert np.shape(dict[axis_direction_name])[1] == 2
		assert type(dict[axis_ratio_name]) == np.ndarray
		assert len(np.shape(dict[axis_ratio_name])) == 1
		assert type(dict[line_of_sight_index_name]) == int
		assert (dict[line_of_sight_index_name] == 0) or (dict[line_of_sight_index_name] == 1) or (
				dict[line_of_sight_index_name] == 2)
		return

	@staticmethod
	def check_jackknife_max_separation(num_jk, boxsize, max_separation, num_r_bins):
		"""Warns if the maximum separation or the number of r bins is large compared to the jackknife subbox size.

		Parameters
		----------
		num_jk : int
			Number of jackknife realisations.
		boxsize : float
			Size of the simulation box.
		max_separation : float
			Maximum (projected) separation of the measurement.
		num_r_bins : int
			Number of (projected) separation bins.

		"""
		L_box = boxsize / num_jk ** (1. / 3)
		if max_separation > L_box:
			warnings.warn(
				"WARNING: your maximum separation exceeds the size of your jackknife subboxes. This is not recommended. "
				"Please lower the number of jackknife realisations or your maximum separation.")
		if num_r_bins ** (3. / 2) > L_box:
			warnings.warn(
				"WARNING: You have too many r(p) bins for your number of jackknife realisation. This is not recommended. "
				"Please lower the number of r(p) bins or increase your number of jackknife realisations.")
		return

	@staticmethod
	def rename_input_keys(input_dict, name_map):
		"""Returns a shallow copy of a data (or mask) dictionary in which user-defined key names are replaced by the
		internal default names. Keys that do not appear in the name map are kept unchanged, the data itself is not
		copied. If None is given, None is returned.

		Parameters
		----------
		input_dict : dict or NoneType
			Data or mask dictionary with user-defined key names.
		name_map : dict
			Mapping of user-defined key names to internal default names.

		Returns
		-------
		dict or NoneType
			Dictionary with the internal default key names.

		"""
		if input_dict is None:
			return None
		return {name_map.get(key, key): value for key, value in input_dict.items()}
