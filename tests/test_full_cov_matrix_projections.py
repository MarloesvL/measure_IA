import numpy as np


def test_full_cov_matrix_projections(IA_TNG100_99_mock):
	IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set1"], 3,
														   False)  # output is read by create_full_cov_matrix_projections
	IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set2"], 3, False)
	IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set3"], 3, False)
	cov3, cov2_12, cov2_13, cov2_23 = IA_TNG100_99_mock.create_full_cov_matrix_projections("w_g_plus",
																						   ["set1", "set2", "set3"], 3,
																						   True)
	np.testing.assert_array_equal(cov3[0:8, 0:8], cov2_12)
	np.testing.assert_array_equal(cov3[4:12, 4:12], cov2_23)
	np.testing.assert_array_equal(cov3[0:4, 0:4], cov2_13[0:4, 0:4])
	np.testing.assert_array_equal(cov3[8:12, 0:4], cov2_13[4:8, 0:4])
	np.testing.assert_array_equal(cov3[8:12, 8:12], cov2_13[4:8, 4:8])
	np.testing.assert_array_equal(cov3[0:4, 8:12], cov2_13[0:4, 4:8])
	return


if __name__ == '__main__':
	test_full_cov_matrix_projections()
