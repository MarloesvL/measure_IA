import h5py
import numpy as np
from measureia import ReadData


def test_masks_w(IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300.hdf5', 'a')
	Mass = data_file["Mass"][:]
	data_file.close()
	masks = {}
	for key_d in IA_mock_TNG300_n8.data.keys():
		masks[key_d] = Mass > 11.75
	IA_mock_TNG300_n8.measure_xi_w("high", 'both', 8, file_tree_path='./data/processed/', masks=masks)

	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
	wgp = output_wgp.read_cat("high")
	rp_wgp = output_wgp.read_cat("high_rp")
	cov_wgp = output_wgp.read_cat("high_jackknife_cov_8")
	wgg = output_wgg.read_cat("high")
	rp_wgg = output_wgg.read_cat("high_rp")
	cov_wgg = output_wgg.read_cat("high_jackknife_cov_8")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
	a_wgp = available_out_wgp.read_cat("high")
	a_rp_wgp = available_out_wgp.read_cat("high_rp")
	a_cov_wgp = available_out_wgp.read_cat("high_jackknife_cov_8")
	a_wgg = available_out_wgg.read_cat("high")
	a_rp_wgg = available_out_wgg.read_cat("high_rp")
	a_cov_wgg = available_out_wgg.read_cat("high_jackknife_cov_8")

	np.testing.assert_array_equal(wgp, a_wgp)
	np.testing.assert_array_equal(wgg, a_wgg)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_array_equal(cov_wgp, a_cov_wgp)
	np.testing.assert_array_equal(cov_wgg, a_cov_wgg)

	return


def test_masks_m(IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300.hdf5', 'a')
	Mass = data_file["Mass"][:]
	data_file.close()
	masks = {}
	for key_d in IA_mock_TNG300_n8.data.keys():
		masks[key_d] = Mass > 11.75
	IA_mock_TNG300_n8.measure_xi_multipoles("high", 'both', 8, file_tree_path='./data/processed/', masks=masks)

	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
						  data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	wgp = output_wgp.read_cat("high")
	rp_wgp = output_wgp.read_cat("high_r")
	cov_wgp = output_wgp.read_cat("high_jackknife_cov_8")
	wgg = output_wgg.read_cat("high")
	rp_wgg = output_wgg.read_cat("high_r")
	cov_wgg = output_wgg.read_cat("high_jackknife_cov_8")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
								 data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	a_wgp = available_out_wgp.read_cat("high")
	a_rp_wgp = available_out_wgp.read_cat("high_r")
	a_cov_wgp = available_out_wgp.read_cat("high_jackknife_cov_8")
	a_wgg = available_out_wgg.read_cat("high")
	a_rp_wgg = available_out_wgg.read_cat("high_r")
	a_cov_wgg = available_out_wgg.read_cat("high_jackknife_cov_8")

	np.testing.assert_array_equal(wgp, a_wgp)
	np.testing.assert_array_equal(wgg, a_wgg)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_array_equal(cov_wgp, a_cov_wgp)
	np.testing.assert_array_equal(cov_wgg, a_cov_wgg)

	return


if __name__ == '__main__':
	test_masks_w()
	test_masks_m()
