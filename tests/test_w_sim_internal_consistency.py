import h5py
import numpy as np
from src.read_data import ReadData


def test_compare_saved_output(IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	IA_mock_TNG300_n8.measure_xi_w("All_both", 'both', 8, file_tree_path='./data/processed/')
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", "All_both", './data/processed/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_gg/", "All_both", './data/processed/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_rp")
	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
	wgg = output_wgg.read_cat("All_both")
	rp_wgg = output_wgg.read_cat("All_both_rp")
	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "w_g_plus/", "All_both", './data/processed/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "w_gg/", "All_both", './data/processed/')
	a_wgp = available_out_wgp.read_cat("All")
	a_rp_wgp = available_out_wgp.read_cat("All_rp")
	a_cov_wgp = available_out_wgp.read_cat("All_jackknife_cov_8")
	a_wgg = available_out_wgg.read_cat("All")
	a_rp_wgg = available_out_wgg.read_cat("All_rp")
	a_cov_wgg = available_out_wgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_array_equal(wgp, a_wgp)
	np.testing.assert_array_equal(wgg, a_wgg)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_array_equal(cov_wgp, a_cov_wgp)
	np.testing.assert_array_equal(cov_wgg, a_cov_wgg)

	return


def test_gg_gp_both(IA_mock_TNG300_n8):
	IA_mock_TNG300_n8.measure_xi_w("All_gp", 'g+', 8, file_tree_path='./data/processed/')
	IA_mock_TNG300_n8.measure_xi_w("All_gg", 'gg', 8, file_tree_path='./data/processed/')

	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", "All_both", './data/processed/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_gg/", "All_both", './data/processed/')
	wgp_b = output_wgp.read_cat("All_both")
	rp_wgp_b = output_wgp.read_cat("All_both_rp")
	wgg_b = output_wgg.read_cat("All_both")
	rp_wgg_b = output_wgg.read_cat("All_both_rp")
	cov_wgp_b = output_wgp.read_cat("All_both_jackknife_cov_8")
	cov_wgg_b = output_wgg.read_cat("All_both_jackknife_cov_8")

	wgp = output_wgp.read_cat("All_gp")
	rp_wgp = output_wgp.read_cat("All_gp_rp")
	wgg = output_wgg.read_cat("All_gg")
	rp_wgg = output_wgg.read_cat("All_gg_rp")
	cov_wgp = output_wgp.read_cat("All_gp_jackknife_cov_8")
	cov_wgg = output_wgg.read_cat("All_gg_jackknife_cov_8")

	print(cov_wgg)
	print(cov_wgg_b)

	np.testing.assert_array_equal(wgp, wgp_b)
	np.testing.assert_array_equal(rp_wgp, rp_wgp_b)
	np.testing.assert_array_equal(wgg, wgg_b)
	np.testing.assert_array_equal(rp_wgg, rp_wgg_b)
	np.testing.assert_array_equal(cov_wgp, cov_wgp_b)
	np.testing.assert_array_equal(cov_wgg, cov_wgg_b)

	return


if __name__ == '__main__':
	test_compare_saved_output()
	test_gg_gp_both()
