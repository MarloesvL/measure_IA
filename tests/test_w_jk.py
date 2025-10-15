import numpy as np
from measureia import ReadData


def test_combine_jk(IA_available_output):
	covs, stds = IA_available_output._combine_jackknife_information("All", "", ["w_g_plus", "w_gg"], 8, True)
	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
	a_cov_wgp = available_out_wgp.read_cat("All_jackknife_cov_8")
	a_cov_wgg = available_out_wgg.read_cat("All_jackknife_cov_8")
	np.testing.assert_allclose(covs[1], a_cov_wgg, rtol=1e-5)
	np.testing.assert_allclose(covs[0], a_cov_wgp, rtol=1e-5)
	return


def test_get_jk_regions(jk_regions):
	jk_pos, jk_shape = jk_regions._get_jackknife_region_indices(None, 2)
	assert jk_pos[0] == 0
	assert jk_pos[1] == 5
	assert jk_pos[2] == 7
	assert jk_pos[3] == 3
	np.testing.assert_array_equal(jk_pos, jk_shape)
	return


def test_compare_saved_output(IA_mock_TNG300_jk):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	IA_mock_TNG300_jk.measure_xi_w_jk("All_both", 'both', 8, file_tree_path=False)
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "w_gg/", data_path='./data/processed/TNG300/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_rp")
	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
	wgg = output_wgg.read_cat("All_both")
	rp_wgg = output_wgg.read_cat("All_both_rp")
	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
	a_wgp = available_out_wgp.read_cat("All")
	a_rp_wgp = available_out_wgp.read_cat("All_rp")
	a_cov_wgp = available_out_wgp.read_cat("All_jackknife_cov_8")
	a_wgg = available_out_wgg.read_cat("All")
	a_rp_wgg = available_out_wgg.read_cat("All_rp")
	a_cov_wgg = available_out_wgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_allclose(wgp, a_wgp, rtol=1e-5)
	np.testing.assert_allclose(wgg, a_wgg, rtol=1e-5)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_allclose(cov_wgg, a_cov_wgg, rtol=1e-5)
	np.testing.assert_allclose(cov_wgp, a_cov_wgp, rtol=1e-5)
	return

# def test_tree_version(IA_mock_TNG300_jk_tree):
# 	IA_mock_TNG300_jk_tree.measure_xi_w_jk("All_both", 'both', 8, file_tree_path='./data/processed/')
# 	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
# 	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
# 	wgp = output_wgp.read_cat("All_both")
# 	rp_wgp = output_wgp.read_cat("All_both_rp")
# 	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
# 	wgg = output_wgg.read_cat("All_both")
# 	rp_wgg = output_wgg.read_cat("All_both_rp")
# 	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")
#
# 	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
# 	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
# 	a_wgp = available_out_wgp.read_cat("All")
# 	a_rp_wgp = available_out_wgp.read_cat("All_rp")
# 	a_cov_wgp = available_out_wgp.read_cat("All_jackknife_cov_8")
# 	a_wgg = available_out_wgg.read_cat("All")
# 	a_rp_wgg = available_out_wgg.read_cat("All_rp")
# 	a_cov_wgg = available_out_wgg.read_cat("All_jackknife_cov_8")
#
# 	np.testing.assert_allclose(wgp, a_wgp, rtol=1e-5)
# 	np.testing.assert_allclose(wgg, a_wgg, rtol=1e-5)
#
# 	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
# 	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)
#
# 	np.testing.assert_allclose(cov_wgg, a_cov_wgg, rtol=1e-5)
# 	np.testing.assert_allclose(cov_wgp, a_cov_wgp, rtol=1e-5)
# 	return
