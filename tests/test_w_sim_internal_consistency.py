import numpy as np
from measureia import ReadData


def test_compare_saved_output(IA_mock_TNG300_n1):
	'''
	Compare limited TNG300 catalogue outputs with saved version.
	:param IA_mock_TNG300:
	:return:
	'''
	IA_mock_TNG300_n1.measure_xi_w("All_both", 'both', 0, temp_file_path='./data/processed/')
	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
	wgp = output_wgp.read_cat("All_both")
	rp_wgp = output_wgp.read_cat("All_both_rp")
	wgg = output_wgg.read_cat("All_both")
	rp_wgg = output_wgg.read_cat("All_both_rp")

	available_out_wgp = ReadData("TNG300", "mock_IA_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	available_out_wgg = ReadData("TNG300", "mock_IA_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
	a_wgp = available_out_wgp.read_cat("All")
	a_rp_wgp = available_out_wgp.read_cat("All_rp")
	a_wgg = available_out_wgg.read_cat("All")
	a_rp_wgg = available_out_wgg.read_cat("All_rp")

	np.testing.assert_array_equal(wgp, a_wgp)
	np.testing.assert_array_equal(wgg, a_wgg)

	np.testing.assert_array_equal(rp_wgp, a_rp_wgp)
	np.testing.assert_array_equal(rp_wgg, a_rp_wgg)

	return


def test_gg_gp_both(IA_mock_TNG300_n1):
	IA_mock_TNG300_n1.measure_xi_w("All_gp", 'g+', 0, temp_file_path='./data/processed/')
	IA_mock_TNG300_n1.measure_xi_w("All_gg", 'gg', 0, temp_file_path='./data/processed/')

	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	output_wgg = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_gg/", data_path='./data/processed/TNG300/')
	wgp_b = output_wgp.read_cat("All_both")
	rp_wgp_b = output_wgp.read_cat("All_both_rp")
	wgg_b = output_wgg.read_cat("All_both")
	rp_wgg_b = output_wgg.read_cat("All_both_rp")

	wgp = output_wgp.read_cat("All_gp")
	rp_wgp = output_wgp.read_cat("All_gp_rp")
	wgg = output_wgg.read_cat("All_gg")
	rp_wgg = output_wgg.read_cat("All_gg_rp")

	np.testing.assert_array_equal(wgp, wgp_b)
	np.testing.assert_array_equal(rp_wgp, rp_wgp_b)
	np.testing.assert_array_equal(wgg, wgg_b)
	np.testing.assert_array_equal(rp_wgg, rp_wgg_b)

	return


def test_versions(IA_mock_TNG300_n1, IA_mock_TNG300_n8):
	IA_mock_TNG300_n1.measure_xi_w("brute", 'both', 0, temp_file_path=False)
	IA_mock_TNG300_n1.measure_xi_w("tree", 'both', 0, temp_file_path='./data/processed/')
	IA_mock_TNG300_n8.measure_xi_w("multiproc", 'both', 0, temp_file_path='./data/processed/', chunk_size=100)

	output_wgp = ReadData("TNG300", "test_IA_mock_TNG300", 99, "w_g_plus/", data_path='./data/processed/TNG300/')
	wgp_brute = output_wgp.read_cat("brute")
	wgp_tree = output_wgp.read_cat("tree")
	wgp_multiproc = output_wgp.read_cat("multiproc")

	rp_brute = output_wgp.read_cat("brute_rp")
	rp_tree = output_wgp.read_cat("tree_rp")
	rp_multiproc = output_wgp.read_cat("multiproc_rp")

	np.testing.assert_array_equal(rp_brute, rp_tree)
	np.testing.assert_array_equal(rp_brute, rp_multiproc)

	np.testing.assert_allclose(wgp_brute, wgp_tree)
	np.testing.assert_allclose(wgp_brute, wgp_multiproc)

	return


if __name__ == '__main__':
	test_compare_saved_output()
	test_gg_gp_both()
	test_versions()
