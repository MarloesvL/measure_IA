# import numpy as np
#
#
# def test_velocity(SV_PT4_TNG100_99):
# 	'''
# 	Test for measure_velocities method based on mock data for three galaxies with variation in masses and pos/neg
# 	particle velocities.
# 	:param SV_PT4_TNG100_99:
# 	:return:
# 	'''
# 	velocity = SV_PT4_TNG100_99.measure_velocities()
# 	np.testing.assert_equal(velocity[0], np.array([2., 2., 2.]))
# 	np.testing.assert_equal(velocity[1], np.array([1., 1., 1.]))
# 	np.testing.assert_equal(velocity[2], np.array([-1, 2., 2.]))
# 	return
#
#
# if __name__ == '__main__':
# 	test_velocity()
