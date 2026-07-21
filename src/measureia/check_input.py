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
		Check types and shapes of the entries of the input data dictionary (Box).
	check_type_input_data_lightcone()
		Check types, shapes and coordinate ranges of the input data dictionary (Lightcone).
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
			if not os.access(folder_path, os.W_OK):
				raise PermissionError(f"The output folder for {path} is not writable: {folder_path}")
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
		coordinates = np.asarray(coordinates)
		if coordinates.ndim != 2 or coordinates.shape[1] != 3:
			raise ValueError(
				f"Coordinates must be a 2-D (N, 3) array, got shape {coordinates.shape}.")
		for i in range(coordinates.shape[1]):
			if coordinates[:, i].min() < 0. or coordinates[:, i].max() > boxsize:
				raise ValueError(
					"The coordinates do not agree with the boxsize. They should be in range [0, boxsize]. Check the units.")
		return

	@staticmethod
	def check_type_input_data(dict, names):
		"""Checks types and shapes of the entries of the input data dictionary for MeasureIABox.

		Raises ``TypeError`` for wrong container types and ``ValueError`` for wrong shapes,
		inconsistent lengths or non-finite values (unlike the previous bare ``assert``s, which
		carried no message and were stripped under ``python -O``).

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

		def _ndarray(key):
			if not isinstance(dict[key], np.ndarray):
				raise TypeError(f"'{key}' must be a numpy ndarray, got {type(dict[key]).__name__}.")
			return dict[key]

		pos_d = _ndarray(positions_density_sample_name)
		pos_s = _ndarray(positions_shape_sample_name)
		axis_dir = _ndarray(axis_direction_name)
		axis_ratio = _ndarray(axis_ratio_name)

		if pos_d.ndim != 2 or pos_d.shape[1] != 3:
			raise ValueError(f"'{positions_density_sample_name}' must have shape (N, 3), got {pos_d.shape}.")
		if pos_s.ndim != 2 or pos_s.shape[1] != 3:
			raise ValueError(f"'{positions_shape_sample_name}' must have shape (N, 3), got {pos_s.shape}.")
		if axis_dir.ndim != 2 or axis_dir.shape[1] != 2:
			raise ValueError(f"'{axis_direction_name}' must have shape (M, 2), got {axis_dir.shape}.")
		if axis_ratio.ndim != 1:
			raise ValueError(f"'{axis_ratio_name}' must be a 1-D array, got shape {axis_ratio.shape}.")

		# axis_direction and axis_ratio describe the shape sample and must align with it
		if len(axis_dir) != len(pos_s):
			raise ValueError(
				f"'{axis_direction_name}' (length {len(axis_dir)}) must match "
				f"'{positions_shape_sample_name}' (length {len(pos_s)}).")
		if len(axis_ratio) != len(pos_s):
			raise ValueError(
				f"'{axis_ratio_name}' (length {len(axis_ratio)}) must match "
				f"'{positions_shape_sample_name}' (length {len(pos_s)}).")

		for key, arr in ((positions_density_sample_name, pos_d), (positions_shape_sample_name, pos_s),
						 (axis_direction_name, axis_dir), (axis_ratio_name, axis_ratio)):
			if not np.isfinite(arr).all():
				raise ValueError(f"'{key}' contains NaN or infinite values.")

		los = dict[line_of_sight_index_name]
		if not isinstance(los, (int, np.integer)):
			raise TypeError(
				f"'{line_of_sight_index_name}' must be an integer (0, 1 or 2), got {type(los).__name__}.")
		if los not in (0, 1, 2):
			raise ValueError(f"'{line_of_sight_index_name}' must be 0, 1 or 2, got {los}.")
		return

	@staticmethod
	def check_type_input_data_lightcone(dict, names):
		"""Checks types, shapes and ranges of the input data dictionary for MeasureIALightcone.

		The lightcone analogue of ``check_type_input_data``: RA/DEC/redshift/e1/e2 must be 1-D
		finite ndarrays; the density-sample coordinates (RA/DEC/redshift) share one length and
		the shape-sample coordinates (RA/DEC/redshift/e1/e2) another (the two samples may
		differ in size); RA lies in [0, 360] and DEC in [-90, 90] degrees.

		Parameters
		----------
		dict : dict
			Data dictionary to check.
		names : iterable of 8 str
			Key names ``(RA_density, RA_shape, DEC_density, DEC_shape, redshift_density,
			redshift_shape, e1, e2)``, in that order.

		"""
		(ra_d, ra_s, dec_d, dec_s, z_d, z_s, e1, e2) = names
		for key in names:
			if not isinstance(dict[key], np.ndarray):
				raise TypeError(f"'{key}' must be a numpy ndarray, got {type(dict[key]).__name__}.")
			if dict[key].ndim != 1:
				raise ValueError(f"'{key}' must be a 1-D array, got shape {dict[key].shape}.")
			if not np.isfinite(dict[key]).all():
				raise ValueError(f"'{key}' contains NaN or infinite values.")

		n_density = len(dict[ra_d])
		for key in (dec_d, z_d):
			if len(dict[key]) != n_density:
				raise ValueError(
					f"'{key}' (length {len(dict[key])}) must match '{ra_d}' (length {n_density}).")
		n_shape = len(dict[ra_s])
		for key in (dec_s, z_s, e1, e2):
			if len(dict[key]) != n_shape:
				raise ValueError(
					f"'{key}' (length {len(dict[key])}) must match '{ra_s}' (length {n_shape}).")

		for key in (ra_d, ra_s):
			if dict[key].min() < 0. or dict[key].max() > 360.:
				raise ValueError(f"'{key}' must be in [0, 360] degrees.")
		for key in (dec_d, dec_s):
			if dict[key].min() < -90. or dict[key].max() > 90.:
				raise ValueError(f"'{key}' must be in [-90, 90] degrees.")
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
