import h5py
import numpy as np
from src.read_data import ReadData


def test_weights_w(IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	IA_mock_TNG300_n8.measure_xi_w("All_both", 'both', 8, file_tree_path='./data/processed/')
	IA_mock_TNG300_n8.data["weight"] = np.array([0.5] * len(IA_mock_TNG300_n8.data["Position"][:, 0]))
	IA_mock_TNG300_n8.data["weight_shape_sample"] = np.array(
		[0.5] * len(IA_mock_TNG300_n8.data["Position_shape_sample"][:, 0]))
	IA_mock_TNG300_n8.measure_xi_w("All_both_weight", 'both', 8, file_tree_path='./data/processed/')

	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", data_path='./data/processed/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_gg/", data_path='./data/processed/')
	output_xigg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w/xi_gg/", data_path='./data/processed/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_rp")
	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
	DD = output_xigg.read_cat("All_both_DD")
	rp_wgg = output_wgg.read_cat("All_both_rp")
	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")

	a_wgp = output_wgp.read_cat("All_both_weight")
	a_rp_wgp = output_wgp.read_cat("All_both_weight_rp")
	a_cov_wgp = output_wgp.read_cat("All_both_weight_jackknife_cov_8")
	a_DD = output_xigg.read_cat("All_both_weight_DD")
	a_rp_wgg = output_wgg.read_cat("All_both_weight_rp")
	a_cov_wgg = output_wgg.read_cat("All_both_weight_jackknife_cov_8")

	np.testing.assert_array_equal(wgp, 4 * a_wgp)
	np.testing.assert_array_equal(DD, 4 * a_DD)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_array_equal(cov_wgp, 16 * a_cov_wgp)
	np.testing.assert_allclose(cov_wgg, 16 * a_cov_wgg, rtol=1e-9)

	return


def test_weights_m(IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	IA_mock_TNG300_n8.measure_xi_multipoles("All_both", 'both', 8, file_tree_path='./data/processed/')
	IA_mock_TNG300_n8.data["weight"] = np.array([0.5] * len(IA_mock_TNG300_n8.data["Position"][:, 0]))
	IA_mock_TNG300_n8.data["weight_shape_sample"] = np.array(
		[0.5] * len(IA_mock_TNG300_n8.data["Position_shape_sample"][:, 0]))
	IA_mock_TNG300_n8.measure_xi_multipoles("All_both_weight", 'both', 8, file_tree_path='./data/processed/')

	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/", data_path='./data/processed/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/", data_path='./data/processed/')
	output_xigg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_gg/", data_path='./data/processed/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_r")
	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
	DD = output_xigg.read_cat("All_both_DD")
	rp_wgg = output_wgg.read_cat("All_both_r")
	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")

	a_wgp = output_wgp.read_cat("All_both_weight")
	a_rp_wgp = output_wgp.read_cat("All_both_weight_r")
	a_cov_wgp = output_wgp.read_cat("All_both_weight_jackknife_cov_8")
	a_DD = output_xigg.read_cat("All_both_weight_DD")
	a_rp_wgg = output_wgg.read_cat("All_both_weight_r")
	a_cov_wgg = output_wgg.read_cat("All_both_weight_jackknife_cov_8")

	np.testing.assert_array_equal(wgp, 4 * a_wgp)
	np.testing.assert_array_equal(DD, 4 * a_DD)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_array_equal(cov_wgp, 16 * a_cov_wgp)
	np.testing.assert_allclose(cov_wgg, 16 * a_cov_wgg, rtol=1e-9)

	return


if __name__ == '__main__':
	test_weights_w()
	test_weights_m()
