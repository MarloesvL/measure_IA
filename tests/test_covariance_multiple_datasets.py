import numpy as np


def test_covariance_multiple_datasets(IA_TNG100_99_mock):
	'''
	:param :
	:return:
	'''
	cov1, std = IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set1"], 3, True)
	cov2, std = IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set2"], 3, True)
	cov3, std = IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set3"], 3, True)
	cov_comb12, std = IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set1", "set2"], 3, True)
	cov_comb13, std = IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set1", "set3"], 3, True)
	cov_comb23, std = IA_TNG100_99_mock.measure_covariance_multiple_datasets("w_g_plus", ["set2", "set3"], 3, True)
	np.testing.assert_array_equal(cov1, cov2)
	np.testing.assert_array_equal(cov2 * 4, cov3)
	np.testing.assert_array_equal(
		cov1, 2 / 3 * np.array([[2, 1, 1, -1], [1, 2, 2, -2], [1, 2, 2, -2], [-1, -2, -2, 2]]))
	# cov (*2/3)
	#  2	1	1	-1
	#  1	2	2	-2
	#  1	2	2	-2
	# -1	-2	-2	2
	np.testing.assert_array_equal(
		cov_comb12, 2 / 3 * np.array([[1, -1, -1, 1], [-1, -2, -2, 2], [-1, -2, -2, 2], [1, 2, 2, -2]]))
	# comb cov (*2/3)
	# 1	-1	-1	1
	# -1-2	-2	2
	# -1 -2	-2	2
	# 1	2	2	-2
	np.testing.assert_array_equal(
		cov_comb13, 2*cov_comb12)
	# 2	-2	-2	2
	# -2 -4	-4	4
	# -2 -4	-4	4
	# 2	4	4	-4
	np.testing.assert_array_equal(
		cov_comb23, 2 * cov1)
	#  4	2	2	-2
	#  2	4	4	-4
	#  2	4	4	-4
	# -2	-4	-4	4

	return


if __name__ == '__main__':
	test_covariance_multiple_datasets()
