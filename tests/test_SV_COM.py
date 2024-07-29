import numpy as np


# np.testing.assert_equal (arrays exact same)
# assert_allclose
# en meer

# unittest - meer opties voor specific cases


# def test_example(sim):
# 	assert True == 1
# 	# assert sim == 16 # see conftest file
# 	return

# need mock data
def test_COM(SV_PT4_TNG100_99):
	COM = SV_PT4_TNG100_99.measure_COM()
	assert max(COM[:, 0]) < SV_PT4_TNG100_99.boxsize
	assert max(COM[:, 1]) < SV_PT4_TNG100_99.boxsize
	assert max(COM[:, 2]) < SV_PT4_TNG100_99.boxsize
	assert min(COM[:, 0]) > 0.0
	assert min(COM[:, 1]) > 0.0
	assert min(COM[:, 2]) > 0.0
	np.testing.assert_equal(COM[0], np.array([2000, 2000, 2000]))
	np.testing.assert_equal(COM[1], np.array([1000, 1000, 1000]))
	np.testing.assert_equal(COM[2], np.array([74000, 2000, 2000]))
	return


if __name__ == '__main__':
	test_COM()
