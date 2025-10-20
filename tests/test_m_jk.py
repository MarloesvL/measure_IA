import numpy as np
from measureia import ReadData


def test_combine_jk(IA_available_output):
	covs, stds = IA_available_output._combine_jackknife_information("All", "", ["multipoles_g_plus", "multipoles_gg"],
																	8, True)
	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
								 data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
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


def test_compare_saved_output_realisations_DD(IA_mock_TNG300_jk, IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	IA_mock_TNG300_jk.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path=False)
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles/xi_gg/All_both_jk8/",
						  data_path='./data/processed/TNG300/')

	IA_mock_TNG300_n8.measure_xi_multipoles("All_both", 'both', 8, file_tree_path='./data/processed/',
											save_jk_terms=True)
	available_out_xigg_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_gg/All_both_jk8/",
									 data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		DD = output_wgg.read_cat(f"All_both_{i}_DD")
		a_DD = available_out_xigg_jk.read_cat(f"All_both_{i}_DD")
		np.testing.assert_allclose(DD, a_DD, rtol=1e-5)
	return


def test_compare_saved_output_realisations_SplusD(IA_mock_TNG300_jk, IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	IA_mock_TNG300_jk.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path=False)
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles/xi_g_plus/All_both_jk8/",
						  data_path='./data/processed/TNG300/')

	IA_mock_TNG300_n8.measure_xi_multipoles("All_both", 'both', 8, file_tree_path='./data/processed/',
											save_jk_terms=True)
	available_out_xi_gg_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/All_both_jk8/",
									  data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		SplusD = output_wgp.read_cat(f"All_both_{i}_SplusD")
		a_SplusD = available_out_xi_gg_jk.read_cat(f"All_both_{i}_SplusD")
		np.testing.assert_allclose(SplusD, a_SplusD, rtol=1e-5, atol=1e-5)  # account for R in samples
	return


def test_compare_saved_output_realisations_RR(IA_mock_TNG300_jk, IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	IA_mock_TNG300_jk.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path=False)
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles/xi_g_plus/All_both_jk8/",
						  data_path='./data/processed/TNG300/')

	IA_mock_TNG300_n8.measure_xi_multipoles("All_both", 'both', 8, file_tree_path='./data/processed/',
											save_jk_terms=True)
	available_out_xi_gg_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/All_both_jk8/",
									  data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		RR = output_wgp.read_cat(f"All_both_{i}_RR")
		a_RR = available_out_xi_gg_jk.read_cat(f"All_both_{i}_RR_g_plus")
		np.testing.assert_allclose(RR, a_RR, rtol=1e-5, atol=1e-5)  # account for R in samples
	return


def test_compare_saved_output_realisations_xi(IA_mock_TNG300_jk, IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	IA_mock_TNG300_jk.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path=False)
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles/xi_g_plus/All_both_jk8/",
						  data_path='./data/processed/TNG300/')

	IA_mock_TNG300_n8.measure_xi_multipoles("All_both", 'both', 8, file_tree_path='./data/processed/',
											save_jk_terms=True)
	available_out_xi_gg_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/All_both_jk8/",
									  data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		xi = output_wgp.read_cat(f"All_both_{i}")
		a_xi = available_out_xi_gg_jk.read_cat(f"All_both_{i}")
		np.testing.assert_allclose(xi, a_xi, rtol=1e-5, atol=1e-5)  # account for R in samples
	return


def test_compare_saved_output_realisations(IA_mock_TNG300_jk, IA_mock_TNG300_n8):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	IA_mock_TNG300_jk.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path=False)
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles_g_plus/All_both_jk8/",
						  data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles_gg/All_both_jk8/",
						  data_path='./data/processed/TNG300/')

	IA_mock_TNG300_n8.measure_xi_multipoles("All_both", 'both', 8, file_tree_path='./data/processed/',
											save_jk_terms=True)
	available_out_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/All_both_jk8/",
								 data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/All_both_jk8/",
								 data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		wgp = output_wgp.read_cat(f"All_both_{i}")
		wgg = output_wgg.read_cat(f"All_both_{i}")
		a_wgp = available_out_wgp.read_cat(f"All_both_{i}")
		a_wgg = available_out_wgg.read_cat(f"All_both_{i}")
		# print(wgg / a_wgg)
		# print(wgp / a_wgp)
		np.testing.assert_allclose(wgg, a_wgg, rtol=1e-5)
		np.testing.assert_allclose(wgp, a_wgp, rtol=1e-5)

	return


def test_compare_saved_output(IA_mock_TNG300_jk):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	IA_mock_TNG300_jk.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path=False)
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles_g_plus/",
						  data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300_jk", 99, "multipoles_gg/",
						  data_path='./data/processed/TNG300/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_r")
	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
	wgg = output_wgg.read_cat("All_both")
	rp_wgg = output_wgg.read_cat("All_both_r")
	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
								 data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	a_wgp = available_out_wgp.read_cat("All")
	a_rp_wgp = available_out_wgp.read_cat("All_r")
	a_cov_wgp = available_out_wgp.read_cat("All_jackknife_cov_8")
	a_wgg = available_out_wgg.read_cat("All")
	a_rp_wgg = available_out_wgg.read_cat("All_r")
	a_cov_wgg = available_out_wgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_allclose(wgp, a_wgp, rtol=1e-5)
	np.testing.assert_allclose(wgg, a_wgg, rtol=1e-5)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_allclose(cov_wgg, a_cov_wgg, rtol=1e-5)
	np.testing.assert_allclose(cov_wgp, a_cov_wgp, rtol=1e-5)
	return


def test_tree_version(IA_mock_TNG300_jk):
	IA_mock_TNG300_jk.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path='./data/processed/')
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
						  data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_r")
	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
	wgg = output_wgg.read_cat("All_both")
	rp_wgg = output_wgg.read_cat("All_both_r")
	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
								 data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	a_wgp = available_out_wgp.read_cat("All")
	a_rp_wgp = available_out_wgp.read_cat("All_r")
	a_cov_wgp = available_out_wgp.read_cat("All_jackknife_cov_8")
	a_wgg = available_out_wgg.read_cat("All")
	a_rp_wgg = available_out_wgg.read_cat("All_r")
	a_cov_wgg = available_out_wgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_allclose(wgp, a_wgp, rtol=1e-5)
	np.testing.assert_allclose(wgg, a_wgg, rtol=1e-5)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_allclose(cov_wgg, a_cov_wgg, rtol=1e-5)
	np.testing.assert_allclose(cov_wgp, a_cov_wgp, rtol=1e-5)
	return


def test_multiproc_version(IA_mock_TNG300_jk_n8):
	IA_mock_TNG300_jk_n8.measure_xi_multipoles_jk("All_both", 'both', 8, file_tree_path='./data/processed/')
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
						  data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_r")
	cov_wgp = output_wgp.read_cat("All_both_jackknife_cov_8")
	wgg = output_wgg.read_cat("All_both")
	rp_wgg = output_wgg.read_cat("All_both_r")
	cov_wgg = output_wgg.read_cat("All_both_jackknife_cov_8")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
								 data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	a_wgp = available_out_wgp.read_cat("All")
	a_rp_wgp = available_out_wgp.read_cat("All_r")
	a_cov_wgp = available_out_wgp.read_cat("All_jackknife_cov_8")
	a_wgg = available_out_wgg.read_cat("All")
	a_rp_wgg = available_out_wgg.read_cat("All_r")
	a_cov_wgg = available_out_wgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_allclose(wgp, a_wgp, rtol=1e-5)
	np.testing.assert_allclose(wgg, a_wgg, rtol=1e-5)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	np.testing.assert_allclose(cov_wgg, a_cov_wgg, rtol=1e-5)
	np.testing.assert_allclose(cov_wgp, a_cov_wgp, rtol=1e-5)
	return
