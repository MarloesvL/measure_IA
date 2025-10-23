import numpy as np
from measureia import ReadData


def test_compare_saved_output_realisations_DD(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	IA_mock_TNG300_n1.measure_xi_multipoles("tree_jk", 'both', 8, temp_file_path='./data/processed/')
	output_multipolesgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_gg/tree_jk_jk8/",
								   data_path='./data/processed/TNG300/')

	IA_mock_TNG300_n1.measure_xi_multipoles("brute_jk", 'both', 8, temp_file_path=False)
	available_out_xigg_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_gg/brute_jk_jk8/",
									 data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		DD = output_multipolesgg.read_cat(f"tree_jk_{i}_DD")
		a_DD = available_out_xigg_jk.read_cat(f"brute_jk_{i}_DD")
		np.testing.assert_allclose(DD, a_DD, rtol=1e-5)
	return


def test_compare_saved_output_realisations_SplusD(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	# IA_mock_TNG300_n1.measure_xi_multipoles("tree_jk", 'both', 8, temp_file_path='./data/processed/')
	output_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/tree_jk_jk8/",
								   data_path='./data/processed/TNG300/')

	# IA_mock_TNG300_n1.measure_xi_multipoles("brute_jk", 'both', 9, temp_file_path=False)
	available_out_xigp_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/brute_jk_jk8/",
									 data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		SplusD = output_multipolesgp.read_cat(f"tree_jk_{i}_SplusD")
		a_SplusD = available_out_xigp_jk.read_cat(f"brute_jk_{i}_SplusD")
		np.testing.assert_allclose(SplusD, a_SplusD, rtol=1e-5, atol=1e-5)  # account for R in samples
	return


def test_compare_saved_output_realisations_RR(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	# IA_mock_TNG300_n1.measure_xi_multipoles("tree_jk", 'both', 8, temp_file_path='./data/processed/')
	output_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/tree_jk_jk8/",
								   data_path='./data/processed/TNG300/')

	# IA_mock_TNG300_n1.measure_xi_multipoles("brute_jk", 'both', 9, temp_file_path=False)
	available_out_xigp_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/brute_jk_jk8/",
									 data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		RR = output_multipolesgp.read_cat(f"tree_jk_{i}_RR")
		a_RR = available_out_xigp_jk.read_cat(f"brute_jk_{i}_RR")
		np.testing.assert_allclose(RR, a_RR, rtol=1e-5, atol=1e-5)  # account for R in samples
	return


def test_compare_saved_output_realisations_xi(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	# IA_mock_TNG300_n1.measure_xi_multipoles("tree_jk", 'both', 8, temp_file_path='./data/processed/')
	output_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/tree_jk_jk8/",
								   data_path='./data/processed/TNG300/')

	# IA_mock_TNG300_n1.measure_xi_multipoles("brute_jk", 'both', 9, temp_file_path=False)
	available_out_xigp_jk = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles/xi_g_plus/brute_jk_jk8/",
									 data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		xi = output_multipolesgp.read_cat(f"tree_jk_{i}")
		a_xi = available_out_xigp_jk.read_cat(f"brute_jk_{i}")
		np.testing.assert_allclose(xi, a_xi, rtol=1e-5, atol=1e-5)  # account for R in samples
	return


def test_compare_saved_output_realisations(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''

	# IA_mock_TNG300_n1.measure_xi("tree_jk", 'both', 8, temp_file_path='./data/processed/')
	output_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/tree_jk_jk8/",
								   data_path='./data/processed/TNG300/')
	output_multipolesgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/tree_jk_jk8/",
								   data_path='./data/processed/TNG300/')

	# IA_mock_TNG300_n1.measure_xi_multipoles("tree_nojk", 'both', 0, temp_file_path='./data/processed/')
	available_out_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/brute_jk_jk8/",
										  data_path='./data/processed/TNG300/')
	available_out_multipolesgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/brute_jk_jk8/",
										  data_path='./data/processed/TNG300/')

	for i in np.arange(0, 8):
		multipolesgp = output_multipolesgp.read_cat(f"tree_jk_{i}")
		multipolesgg = output_multipolesgg.read_cat(f"tree_jk_{i}")
		a_multipolesgp = available_out_multipolesgp.read_cat(f"brute_jk_{i}")
		a_multipolesgg = available_out_multipolesgg.read_cat(f"brute_jk_{i}")
		np.testing.assert_allclose(multipolesgg, a_multipolesgg, rtol=1e-5)
		np.testing.assert_allclose(multipolesgp, a_multipolesgp, rtol=1e-5)

	return


def test_compare_saved_output(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	# IA_mock_TNG300_n1.measure_xi_multipoles("tree_jk", 'both', 8, temp_file_path='./data/processed/')
	output_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
								   data_path='./data/processed/TNG300/')
	output_multipolesgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/",
								   data_path='./data/processed/TNG300/')
	multipolesgp = output_multipolesgp.read_cat("tree_jk")
	r_multipolesgp = output_multipolesgp.read_cat("tree_jk_r")
	cov_multipolesgp = output_multipolesgp.read_cat("tree_jk_jackknife_cov_8")
	multipolesgg = output_multipolesgg.read_cat("tree_jk")
	r_multipolesgg = output_multipolesgg.read_cat("tree_jk_r")
	cov_multipolesgg = output_multipolesgg.read_cat("tree_jk_jackknife_cov_8")

	available_out_multipolesgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
										  data_path='./data/processed/TNG300/')
	available_out_multipolesgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/",
										  data_path='./data/processed/TNG300/')
	a_multipolesgp = available_out_multipolesgp.read_cat("All")
	a_r_multipolesgp = available_out_multipolesgp.read_cat("All_r")
	a_cov_multipolesgp = available_out_multipolesgp.read_cat("All_jackknife_cov_8")
	a_multipolesgg = available_out_multipolesgg.read_cat("All")
	a_r_multipolesgg = available_out_multipolesgg.read_cat("All_r")
	a_cov_multipolesgg = available_out_multipolesgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_allclose(multipolesgp, a_multipolesgp, rtol=1e-5)
	np.testing.assert_allclose(multipolesgg, a_multipolesgg, rtol=1e-5)

	np.testing.assert_array_equal(r_multipolesgp, a_r_multipolesgp)
	np.testing.assert_array_equal(r_multipolesgg, a_r_multipolesgg)

	np.testing.assert_allclose(cov_multipolesgg, a_cov_multipolesgg, rtol=1e-5)
	np.testing.assert_allclose(cov_multipolesgp, a_cov_multipolesgp, rtol=1e-5)
	return


def test_tree_version(IA_mock_TNG300_n1):
	# IA_mock_TNG300_n1.measure_xi_multipoles("tree_jk", 'both', 8, temp_file_path='./data/processed/')
	output_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
								   data_path='./data/processed/TNG300/')
	output_multipolesgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/",
								   data_path='./data/processed/TNG300/')
	multipolesgp = output_multipolesgp.read_cat("tree_jk")
	r_multipolesgp = output_multipolesgp.read_cat("tree_jk_r")
	cov_multipolesgp = output_multipolesgp.read_cat("tree_jk_jackknife_cov_8")
	multipolesgg = output_multipolesgg.read_cat("tree_jk")
	r_multipolesgg = output_multipolesgg.read_cat("tree_jk_r")
	cov_multipolesgg = output_multipolesgg.read_cat("tree_jk_jackknife_cov_8")

	available_out_multipolesgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
										  data_path='./data/processed/TNG300/')
	available_out_multipolesgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/",
										  data_path='./data/processed/TNG300/')
	a_multipolesgp = available_out_multipolesgp.read_cat("All")
	a_r_multipolesgp = available_out_multipolesgp.read_cat("All_r")
	a_cov_multipolesgp = available_out_multipolesgp.read_cat("All_jackknife_cov_8")
	a_multipolesgg = available_out_multipolesgg.read_cat("All")
	a_r_multipolesgg = available_out_multipolesgg.read_cat("All_r")
	a_cov_multipolesgg = available_out_multipolesgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_allclose(multipolesgp, a_multipolesgp, rtol=1e-5)
	np.testing.assert_allclose(multipolesgg, a_multipolesgg, rtol=1e-5)

	np.testing.assert_array_equal(r_multipolesgp, a_r_multipolesgp)
	np.testing.assert_array_equal(r_multipolesgg, a_r_multipolesgg)

	np.testing.assert_allclose(cov_multipolesgg, a_cov_multipolesgg, rtol=1e-5)
	np.testing.assert_allclose(cov_multipolesgp, a_cov_multipolesgp, rtol=1e-5)
	return


def test_multiproc_version(IA_mock_TNG300_n8):
	IA_mock_TNG300_n8.measure_xi_multipoles("multiproc_jk", 'both', 8, temp_file_path='./data/processed/',
											chunk_size=100)
	output_multipolesgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
								   data_path='./data/processed/TNG300/')
	output_multipolesgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/",
								   data_path='./data/processed/TNG300/')
	multipolesgp = output_multipolesgp.read_cat("multiproc_jk")
	r_multipolesgp = output_multipolesgp.read_cat("multiproc_jk_r")
	cov_multipolesgp = output_multipolesgp.read_cat("multiproc_jk_jackknife_cov_8")
	multipolesgg = output_multipolesgg.read_cat("multiproc_jk")
	r_multipolesgg = output_multipolesgg.read_cat("multiproc_jk_r")
	cov_multipolesgg = output_multipolesgg.read_cat("multiproc_jk_jackknife_cov_8")

	available_out_multipolesgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
										  data_path='./data/processed/TNG300/')
	available_out_multipolesgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/",
										  data_path='./data/processed/TNG300/')
	a_multipolesgp = available_out_multipolesgp.read_cat("All")
	a_r_multipolesgp = available_out_multipolesgp.read_cat("All_r")
	a_cov_multipolesgp = available_out_multipolesgp.read_cat("All_jackknife_cov_8")
	a_multipolesgg = available_out_multipolesgg.read_cat("All")
	a_r_multipolesgg = available_out_multipolesgg.read_cat("All_r")
	a_cov_multipolesgg = available_out_multipolesgg.read_cat("All_jackknife_cov_8")

	np.testing.assert_allclose(multipolesgp, a_multipolesgp, rtol=1e-5)
	np.testing.assert_allclose(multipolesgg, a_multipolesgg, rtol=1e-5)

	np.testing.assert_array_equal(r_multipolesgp, a_r_multipolesgp)
	np.testing.assert_array_equal(r_multipolesgg, a_r_multipolesgg)

	np.testing.assert_allclose(cov_multipolesgg, a_cov_multipolesgg, rtol=1e-5)
	np.testing.assert_allclose(cov_multipolesgp, a_cov_multipolesgp, rtol=1e-5)
	return


if __name__ == "__main__":
	pass
