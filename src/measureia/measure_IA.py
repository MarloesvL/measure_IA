import sympy
import numpy as np
from .measure_w_box_jk import MeasureWBoxJackknife
from .measure_m_box_jk import MeasureMBoxJackknife
from .measure_w_box import MeasureWBox
from .measure_m_box import MeasureMultipolesBox
from .measure_jackknife import MeasureJackknife
from .measure_galaxy_box import MeasureGalaxyContributionsBox
from .check_input import CheckInput
from . import worker_pool


class MeasureIABox(MeasureWBox, MeasureMultipolesBox, MeasureWBoxJackknife, MeasureMBoxJackknife, MeasureJackknife,
				   MeasureGalaxyContributionsBox, CheckInput):
	r"""Manages the IA correlation function measurement methods used in the MeasureIA package based on speed and input.
	This class is used to call the methods that measure $w_{gg}$, $w_{g+}$ and multipoles for simulations in cartesian
	coordinates. Depending on the input parameters, various correlations incl covariance estimates are measured for
	given data.

	Methods
	-------
	measure_xi_w()
		Compute projected correlations $w_{gg}$ and/or $w_{g+}$.
	measure_xi_multipoles()
		Compute multipoles of the correlation functions, $\tilde{\xi}_{gg,0}$ and/or $\tilde{\xi}_{g+,2}$.

	Notes
	-----
	Inherits attributes from 'SimInfo', where 'boxsize', 'L_0p5' and 'snap_group' are used in this class.
	Inherits attributes from 'MeasureIABase', where 'data', 'output_file_name', 'periodicity', 'Num_position',
	'Num_shape', 'r_min', 'r_max', 'num_bins_r', 'num_bins_pi', 'r_bins', 'pi_bins', 'mu_r_bins' are used.

	"""

	def __init__(
			self,
			data,
			output_file_name,
			simulation=None,
			snapshot=None,
			separation_limits=[0.1, 20.0],
			num_bins_r=8,
			num_bins_pi=20,
			pi_max=None,
			boxsize=None,
			periodicity=True,
			num_nodes=1,
			positions_density_sample_name="Position",
			positions_shape_sample_name="Position_shape_sample",
			axis_direction_name="Axis_Direction",
			axis_ratio_name="q",
			line_of_sight_index_name="LOS",
			weight_density_sample_name="weight",
			weight_shape_sample_name="weight_shape_sample",
			num_overlap=None,
	):
		"""
		The __init__ method of the MeasureIABox class.

		Parameters
		----------
		num_nodes : int, optional
			Number of cores to be used in multiprocessing. Default is 1.
		positions_density_sample_name : str, optional
			Name of the key in the data dictionary that contains the positions of the density sample.
		positions_shape_sample_name : str, optional
			Name of the key in the data dictionary that contains the positions of the shape sample.
		axis_direction_name : str, optional
			Name of the key in the data dictionary that contains the axis direction vectors of the shape sample.
		axis_ratio_name : str, optional
			Name of the key in the data dictionary that contains the axis ratios of the shape sample.
		line_of_sight_index_name : str, optional
			Name of the key in the data dictionary that contains the column index of the line of sight in the
			position vectors.
		weight_density_sample_name : str, optional
			Name of the key in the data dictionary that contains the weights of the density sample.
		weight_shape_sample_name : str, optional
			Name of the key in the data dictionary that contains the weights of the shape sample.
		num_overlap : int or NoneType, optional
			Number of objects present in *both* the position and the shape sample. The analytic
			RR is normalised by ``Num_position * Num_shape - num_overlap``, because a shape
			galaxy cannot pair with itself and the pair loop already drops that self-pair (the
			separation window starts at ``r_min > 0``). Default None, which measures the overlap
			from the coordinates and is what you want for real data, where the shape sample is
			normally drawn from the position sample. Pass an integer to override it -- most
			usefully ``0``, which reproduces the convention external codes such as halotools
			and corr_pc use, where the two samples are treated as independent. An override is
			applied uniformly, so the per-jackknife-region adjustment is only made in the
			default (measured) mode.

		Notes
		-----
		Constructor parameters 'data', 'output_file_name', 'simulation', 'snapshot', 'separation_limits', 'num_bins_r',
		'num_bins_pi', 'pi_max', 'boxsize' and 'periodicity' are passed to MeasureIABase.
		The data dictionary (and any mask dictionaries passed to the measurement methods) may use any key names;
		they are given through the *_name parameters and translated to the internal default names on input.

		"""
		self._input_name_map = {
			positions_density_sample_name: "Position",
			positions_shape_sample_name: "Position_shape_sample",
			axis_direction_name: "Axis_Direction",
			axis_ratio_name: "q",
			line_of_sight_index_name: "LOS",
			weight_density_sample_name: "weight",
			weight_shape_sample_name: "weight_shape_sample",
		}
		if output_file_name is not None:
			self.check_paths([output_file_name])
		if data is not None:
			self.check_dict(data, [positions_density_sample_name, positions_shape_sample_name, axis_direction_name,
								   axis_ratio_name, line_of_sight_index_name])
			self.check_type_input_data(data,
									   (positions_density_sample_name, positions_shape_sample_name, axis_direction_name,
										axis_ratio_name, line_of_sight_index_name))
			data = self.rename_input_keys(data, self._input_name_map)
		super().__init__(data, output_file_name, simulation, snapshot, separation_limits, num_bins_r, num_bins_pi,
						 pi_max, boxsize, periodicity)
		if self.data is not None and self.boxsize is not None:
			self.check_units_coordinates(self.data["Position"], self.boxsize)
		if not (isinstance(num_nodes, (int, np.integer)) and not isinstance(num_nodes, bool) and num_nodes >= 1):
			raise ValueError(f"num_nodes must be an integer >= 1, got {num_nodes!r}.")
		self.num_nodes = num_nodes
		if num_overlap is not None and not (isinstance(num_overlap, (int, np.integer))
											and not isinstance(num_overlap, bool) and num_overlap >= 0):
			raise ValueError(f"num_overlap must be None or an integer >= 0, got {num_overlap!r}.")
		self._num_overlap_override = num_overlap
		self.randoms_data = None
		self.data_dir = None
		self.num_samples = None

		return

	@staticmethod
	def _validate_measure_options(corr_type, ellipticity, num_jk):
		"""Validate the user-facing option strings and ``num_jk`` up front, before any pair
		counting. Previously ``corr_type`` was only checked at the reduction stage (after the
		full count) with a ``KeyError``, ``ellipticity`` only inside the backends, and a
		negative ``num_jk`` was silently treated as 0; all now raise a uniform ``ValueError``."""
		if corr_type not in ("g+", "gg", "both"):
			raise ValueError(f"Unknown corr_type {corr_type!r}. Choose from ['g+', 'gg', 'both'].")
		if ellipticity not in ("distortion", "ellipticity"):
			raise ValueError(
				f"Unknown ellipticity {ellipticity!r}. Choose from ['distortion', 'ellipticity'].")
		if not (isinstance(num_jk, (int, np.integer)) and not isinstance(num_jk, bool) and num_jk >= 0):
			raise ValueError(f"num_jk must be an integer >= 0, got {num_jk!r}.")

	@worker_pool.pooled
	def measure_xi_w(self, dataset_name, corr_type, num_jk=0, temp_file_path=None, masks=None,
					 ellipticity='distortion', chunk_size=1000, responsivity=True):
		r"""Measures $\xi_{gg}$, $\xi_{g+}$ and $w_{gg}$, $w_{g+}$ including jackknife covariance if desired.
		Manages the various _measure_xi_rp_pi_box method options in MeasureWBox and MeasureWBoxJackknife.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		corr_type : str
			Type of correlation to be measured. Choose from [g+, gg, both].
		num_jk : int, optional
			Number of jackknife regions (needs to be x^3, with x an int) for the covariance measurement.
			Default is 0 (no covariance).
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		masks : dict or NoneType, optional
			Directory of mask information in the same form as the data dictionary, where the masks are placed over
			the data to apply selections. Default is None.
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. If larger, more RAM is needed per node.
			Default is 1000.
		ellipticity : str, optional
			Definition of ellipticity. Choose from 'distortion', defined as (1-q^2)/(1+q^2), or 'ellipticity', defined
			 as (1-q)/(1+q). Default is 'distortion'.
		responsivity : bool, optional
			If True (default), the g+ shape signal is calibrated by the responsivity factor 2R, with
			R = <w (1 - e^2 / 2)> / <w> the weighted shear responsivity (R falls back to 0.5, i.e. no correction,
			when False or when the shape weights sum to zero). Use the default for raw shape distortions derived
			from axis ratios; set to False when the input shapes are already calibrated shears. Only the g+
			correlations are affected; the clustering (gg) signal is unchanged. Default is True.

		"""
		self._validate_measure_options(corr_type, ellipticity, num_jk)
		self.responsivity_correction = responsivity
		masks = self.rename_input_keys(masks, self._input_name_map)
		if num_jk > 0:
			try:
				assert sympy.integer_nthroot(num_jk, 3)[1]
				L = sympy.integer_nthroot(num_jk, 3)[0]
				self.check_jackknife_max_separation(num_jk, self.boxsize, self.r_max, self.num_bins_r)
			except AssertionError:
				raise ValueError(
					f"Use x^3 as input for num_jk, with x as an int. {float(int(num_jk ** (1. / 3)))},{num_jk ** (1. / 3)}")

		if temp_file_path == False:
			temp_storage = False
			temp_file_path = None
		else:
			temp_storage = True
		if temp_storage and temp_file_path == None:
			raise ValueError(
				"Input temp_file_path for faster computation. Do not want to save data temporarily? Input file_path_tree=False.")

		if self.data is not None and "RA" in self.data:
			raise TypeError("Given data is lightcone data (contains 'RA'). Use MeasureIALightcone instead.")

		if num_jk > 0:  # include covariance
			if corr_type == "gg":  # DD-only pair counting, skips shape/ellipticity computation
				if self.num_nodes > 1 and temp_storage:
					self._count_pairs_xi_rp_pi_box_jk_multiprocessing(masks=masks, L_subboxes=L,
																	  dataset_name=dataset_name,
																	  return_output=False,
																	  num_nodes=self.num_nodes,
																	  jk_group_name=f"{dataset_name}_jk{num_jk}",
																	  chunk_size=chunk_size,
																	  temp_file_path=temp_file_path)
				elif temp_storage:
					self._count_pairs_xi_rp_pi_box_jk_tree(masks=masks, L_subboxes=L, dataset_name=dataset_name,
														   return_output=False,
														   jk_group_name=f"{dataset_name}_jk{num_jk}")
				else:
					self._count_pairs_xi_rp_pi_box_jk_brute(masks=masks, L_subboxes=L, dataset_name=dataset_name,
															return_output=False,
															jk_group_name=f"{dataset_name}_jk{num_jk}")
			elif self.num_nodes > 1 and temp_storage:
				self._measure_xi_rp_pi_box_jk_multiprocessing(masks=masks, L_subboxes=L, dataset_name=dataset_name,
															  return_output=False,
															  num_nodes=self.num_nodes,
															  jk_group_name=f"{dataset_name}_jk{num_jk}",
															  chunk_size=chunk_size, ellipticity=ellipticity,
															  temp_file_path=temp_file_path)
			elif temp_storage:
				self._measure_xi_rp_pi_box_jk_tree(masks=masks, L_subboxes=L, dataset_name=dataset_name,
												   return_output=False, ellipticity=ellipticity,
												   jk_group_name=f"{dataset_name}_jk{num_jk}")
			else:
				self._measure_xi_rp_pi_box_jk_brute(masks=masks, L_subboxes=L, dataset_name=dataset_name,
													return_output=False, ellipticity=ellipticity,
													jk_group_name=f"{dataset_name}_jk{num_jk}")
			self._measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
			for i in np.arange(num_jk):
				self._measure_w_g_i(corr_type=corr_type, dataset_name=f"{dataset_name}_{i}",
									jk_group_name=f"{dataset_name}_jk{num_jk}", return_output=False)
			if corr_type == "both":
				corr_group = ["w_g_plus", "w_gg"]
			elif corr_type == "g+":
				corr_group = ["w_g_plus"]
			elif corr_type == "gg":
				corr_group = ["w_gg"]
			else:
				raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
			self._combine_jackknife_information(dataset_name=dataset_name, jk_group_name=f"{dataset_name}_jk{num_jk}",
												corr_group=corr_group, num_box=num_jk)
		else:  # no covariance
			if corr_type == "gg":  # DD-only pair counting, skips shape/ellipticity computation
				if self.num_nodes > 1 and temp_storage:
					self._count_pairs_xi_rp_pi_box_multiprocessing(dataset_name=dataset_name,
																   temp_file_path=temp_file_path,
																   masks=masks, return_output=False,
																   num_nodes=self.num_nodes, chunk_size=chunk_size)
				elif temp_storage:
					self._count_pairs_xi_rp_pi_box_tree(masks=masks, dataset_name=dataset_name, return_output=False)
				else:
					self._count_pairs_xi_rp_pi_box_brute(masks=masks, dataset_name=dataset_name, return_output=False)
			elif self.num_nodes > 1 and temp_storage:
				self._measure_xi_rp_pi_box_multiprocessing(dataset_name=dataset_name, temp_file_path=temp_file_path,
														   masks=masks, return_output=False, num_nodes=self.num_nodes,
														   chunk_size=chunk_size, ellipticity=ellipticity)
			elif temp_storage:
				self._measure_xi_rp_pi_box_tree(masks=masks, dataset_name=dataset_name,
												return_output=False, ellipticity=ellipticity)
			else:
				self._measure_xi_rp_pi_box_brute(masks=masks, dataset_name=dataset_name,
												 return_output=False, ellipticity=ellipticity)
			self._measure_w_g_i(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		return

	@worker_pool.pooled
	def measure_xi_multipoles(self, dataset_name, corr_type, num_jk=0, temp_file_path=None, masks=None, rp_cut=None,
							  ellipticity='distortion', chunk_size=1000, responsivity=True):
		r"""Measures $\xi_{gg}$, $\xi_{g+}$ and $\tilde{\xi}_{gg,0}$, $\tilde{\xi}_{g+,2}$ including jackknife covariance
		if desired. Manages the various _measure_xi_r_mur_box method options in MeasureMultipolesBox and
		MeasureMultipolesBoxJackknife.

		Parameters
		----------
		dataset_name : str
			Name of the dataset in the output file.
		corr_type : str
			Type of correlation to be measured. Choose from [g+, gg, both].
		num_jk : int, optional
			Number of jackknife regions (needs to be x^3, with x an int) for the covariance measurement. Default is 0 (no covariance).
		temp_file_path : str or NoneType, optional
			Path to where the data is temporarily stored [file name generated automatically].
		masks : dict or NoneType, optional
			Directory of mask information in the same form as the data dictionary, where the masks are placed over
			the data to apply selections. Default is None.
		rp_cut : float or NoneType, optional
			Applies a minimum r_p value condition for pairs to be included. Default is None.
		chunk_size: int, optional
			Size of the chunks of data sent to each multiprocessing node. If larger, more RAM is needed per node.
		ellipticity : str, optional
			Definition of ellipticity. Choose from 'distortion', defined as (1-q^2)/(1+q^2), or 'ellipticity', defined
			 as (1-q)/(1+q). Default is 'distortion'.
		"""
		self._validate_measure_options(corr_type, ellipticity, num_jk)
		self.responsivity_correction = responsivity
		masks = self.rename_input_keys(masks, self._input_name_map)
		if num_jk > 0:
			try:
				assert sympy.integer_nthroot(num_jk, 3)[1]
				L = sympy.integer_nthroot(num_jk, 3)[0]
				self.check_jackknife_max_separation(num_jk, self.boxsize, self.r_max, self.num_bins_r)
			except AssertionError:
				raise ValueError(
					f"Use x^3 as input for num_jk, with x as an int. {float(int(num_jk ** (1. / 3)))},{num_jk ** (1. / 3)}")

		if temp_file_path == False:
			temp_storage = False
			temp_file_path = None
		else:
			temp_storage = True
		if temp_storage and temp_file_path == None:
			raise ValueError(
				"Input temp_file_path for faster computation. Do not want to save data temporarily? Input file_path_tree=False.")

		if self.data is not None and "RA" in self.data:
			raise TypeError("Given data is lightcone data (contains 'RA'). Use MeasureIALightcone instead.")

		if num_jk > 0:  # include covariance
			if corr_type == "gg":  # DD-only pair counting, skips shape/ellipticity computation
				if self.num_nodes > 1 and temp_storage:
					self._count_pairs_xi_r_mur_box_jk_multiprocessing(masks=masks, L_subboxes=L,
																	  dataset_name=dataset_name,
																	  return_output=False, rp_cut=rp_cut,
																	  num_nodes=self.num_nodes,
																	  jk_group_name=f"{dataset_name}_jk{num_jk}",
																	  chunk_size=chunk_size,
																	  temp_file_path=temp_file_path)
				elif temp_storage:
					self._count_pairs_xi_r_mur_box_jk_tree(masks=masks, L_subboxes=L, dataset_name=dataset_name,
														   return_output=False, rp_cut=rp_cut,
														   jk_group_name=f"{dataset_name}_jk{num_jk}")
				else:
					self._count_pairs_xi_r_mur_box_jk_brute(masks=masks, L_subboxes=L, dataset_name=dataset_name,
															return_output=False, rp_cut=rp_cut,
															jk_group_name=f"{dataset_name}_jk{num_jk}")
			elif self.num_nodes > 1 and temp_storage:
				self._measure_xi_r_mur_box_jk_multiprocessing(masks=masks, L_subboxes=L, dataset_name=dataset_name,
															  return_output=False, rp_cut=rp_cut,
															  num_nodes=self.num_nodes,
															  jk_group_name=f"{dataset_name}_jk{num_jk}",
															  chunk_size=chunk_size, ellipticity=ellipticity,
															  temp_file_path=temp_file_path)
			elif temp_storage:
				self._measure_xi_r_mur_box_jk_tree(masks=masks, L_subboxes=L, dataset_name=dataset_name,
												   return_output=False, rp_cut=rp_cut, ellipticity=ellipticity,
												   jk_group_name=f"{dataset_name}_jk{num_jk}")
			else:
				self._measure_xi_r_mur_box_jk_brute(masks=masks, L_subboxes=L, dataset_name=dataset_name,
													return_output=False, rp_cut=rp_cut, ellipticity=ellipticity,
													jk_group_name=f"{dataset_name}_jk{num_jk}")
			self._measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)
			for i in np.arange(num_jk):
				self._measure_multipoles(corr_type=corr_type, dataset_name=f"{dataset_name}_{i}",
										 jk_group_name=f"{dataset_name}_jk{num_jk}", return_output=False)
			if corr_type == "both":
				corr_group = ["multipoles_g_plus", "multipoles_gg"]
			elif corr_type == "g+":
				corr_group = ["multipoles_g_plus"]
			elif corr_type == "gg":
				corr_group = ["multipoles_gg"]
			else:
				raise KeyError("Unknown value for corr_type. Choose from [g+, gg, both]")
			self._combine_jackknife_information(dataset_name=dataset_name, jk_group_name=f"{dataset_name}_jk{num_jk}",
												corr_group=corr_group, num_box=num_jk)
		else:  # no covariance
			if corr_type == "gg":  # DD-only pair counting, skips shape/ellipticity computation
				if self.num_nodes > 1 and temp_storage:
					self._count_pairs_xi_r_mur_box_multiprocessing(dataset_name=dataset_name,
																   temp_file_path=temp_file_path,
																   masks=masks, return_output=False, rp_cut=rp_cut,
																   num_nodes=self.num_nodes, chunk_size=chunk_size)
				elif temp_storage:
					self._count_pairs_xi_r_mur_box_tree(masks=masks, dataset_name=dataset_name,
														return_output=False, rp_cut=rp_cut)
				else:
					self._count_pairs_xi_r_mur_box_brute(masks=masks, dataset_name=dataset_name,
														 return_output=False, rp_cut=rp_cut)
			elif self.num_nodes > 1 and temp_storage:
				self._measure_xi_r_mur_box_multiprocessing(dataset_name=dataset_name, temp_file_path=temp_file_path,
														   masks=masks, return_output=False, rp_cut=rp_cut,
														   num_nodes=self.num_nodes,
														   chunk_size=chunk_size, ellipticity=ellipticity)
			elif temp_storage:
				self._measure_xi_r_mur_box_tree(masks=masks, dataset_name=dataset_name,
												return_output=False, rp_cut=rp_cut,
												ellipticity=ellipticity)
			else:
				self._measure_xi_r_mur_box_brute(masks=masks, dataset_name=dataset_name,
												 return_output=False, rp_cut=rp_cut, ellipticity=ellipticity)
			self._measure_multipoles(corr_type=corr_type, dataset_name=dataset_name, return_output=False)

		return


if __name__ == "__main__":
	pass
