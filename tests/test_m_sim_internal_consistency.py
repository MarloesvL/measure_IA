import numpy as np
from measureia import ReadData


def test_compare_saved_output(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	IA_mock_TNG300_n1.measure_xi_multipoles("All_both", 'both', 0, temp_file_path='./data/processed/')
	output_mgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
						  data_path='./data/processed/TNG300/')
	output_mgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	mgp = output_mgp.read_cat("All_both")
	r_mgp = output_mgp.read_cat("All_both_r")
	mgg = output_mgg.read_cat("All_both")
	r_mgg = output_mgg.read_cat("All_both_r")

	available_out_mgp = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_g_plus/",
								 data_path='./data/processed/TNG300/')
	available_out_mgg = ReadData("TNG300", "mock_IA_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	a_mgp = available_out_mgp.read_cat("All")
	a_r_mgp = available_out_mgp.read_cat("All_r")
	a_mgg = available_out_mgg.read_cat("All")
	a_r_mgg = available_out_mgg.read_cat("All_r")

	np.testing.assert_array_equal(mgp, a_mgp)
	np.testing.assert_array_equal(mgg, a_mgg)

	np.testing.assert_array_equal(r_mgp, a_r_mgp)
	np.testing.assert_array_equal(r_mgg, a_r_mgg)

	return


def test_gg_gp_both(IA_mock_TNG300_n1):
	IA_mock_TNG300_n1.measure_xi_multipoles("All_gp", 'g+', 0, temp_file_path='./data/processed/')
	IA_mock_TNG300_n1.measure_xi_multipoles("All_gg", 'gg', 0, temp_file_path='./data/processed/')

	output_mgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
						  data_path='./data/processed/TNG300/')
	output_mgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_gg/", data_path='./data/processed/TNG300/')
	mgp_b = output_mgp.read_cat("All_both")
	r_mgp_b = output_mgp.read_cat("All_both_r")
	mgg_b = output_mgg.read_cat("All_both")
	r_mgg_b = output_mgg.read_cat("All_both_r")

	mgp = output_mgp.read_cat("All_gp")
	r_mgp = output_mgp.read_cat("All_gp_r")
	mgg = output_mgg.read_cat("All_gg")
	r_mgg = output_mgg.read_cat("All_gg_r")

	np.testing.assert_array_equal(mgp, mgp_b)
	np.testing.assert_array_equal(r_mgp, r_mgp_b)
	np.testing.assert_array_equal(mgg, mgg_b)
	np.testing.assert_array_equal(r_mgg, r_mgg_b)

	return


def test_versions(IA_mock_TNG300_n1, IA_mock_TNG300_n8):
	IA_mock_TNG300_n1.measure_xi_multipoles("brute", 'both', 0, temp_file_path=False)
	IA_mock_TNG300_n1.measure_xi_multipoles("tree", 'both', 0, temp_file_path='./data/processed/')
	IA_mock_TNG300_n8.measure_xi_multipoles("multiproc", 'both', 0, temp_file_path='./data/processed/', chunk_size=100)

	output_mgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "multipoles_g_plus/",
						  data_path='./data/processed/TNG300/')
	mgp_brute = output_mgp.read_cat("brute")
	mgp_tree = output_mgp.read_cat("tree")
	mgp_multiproc = output_mgp.read_cat("multiproc")

	r_brute = output_mgp.read_cat("brute_r")
	r_tree = output_mgp.read_cat("tree_r")
	r_multiproc = output_mgp.read_cat("multiproc_r")

	np.testing.assert_array_equal(r_brute, r_tree)
	np.testing.assert_array_equal(r_brute, r_multiproc)

	np.testing.assert_allclose(mgp_brute, mgp_tree)
	np.testing.assert_allclose(mgp_brute, mgp_multiproc)

	return


if __name__ == '__main__':
	test_compare_saved_output()
	test_gg_gp_both()
	test_versions()
