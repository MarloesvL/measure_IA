import pytest
import h5py
import numpy as np
from measureia import MeasureIABox


# this file has to be called conftest otherwise the fixtures will not work

# example
# @pytest.fixture()
# def sim():
# 	return demo_sim(4)
#
# def demo_sim(a):
# 	return a**2


@pytest.fixture()
def IA_mock_TNG300_n8():
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300.hdf5', 'a')
	COM = data_file["COM"][:]
	Semimajor_Axis_Direction = data_file["Semimajor_Axis_Direction"][:]
	q = data_file["q"][:]
	data_file.close()
	data_dir = {
		"Position": COM,
		"Position_shape_sample": COM,
		"Axis_Direction": Semimajor_Axis_Direction,
		"LOS": 2,
		"q": q
	}
	return MeasureIABox(data_dir, f"./data/processed/TNG300/test_IA_mock_TNG300.hdf5", "TNG300",
						99, [0.1, 20], 10, 8, None, num_nodes=8)


@pytest.fixture()
def IA_mock_TNG300_n1_large():
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300_large.hdf5', 'a')
	COM = data_file["COM"][:]
	Semimajor_Axis_Direction = data_file["Semimajor_Axis_Direction"][:]
	q = data_file["q"][:]
	data_file.close()
	data_dir = {
		"Position": COM,
		"Position_shape_sample": COM,
		"Axis_Direction": Semimajor_Axis_Direction,
		"LOS": 2,
		"q": q
	}
	return MeasureIABox(data_dir, f"./data/processed/TNG300/test_IA_mock_TNG300_large.hdf5",
					 "TNG300", 99, [0.1, 20], 10,
						8, None, num_nodes=1)


@pytest.fixture()
def IA_mock_TNG300_n8_large():
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300_large.hdf5', 'a')
	COM = data_file["COM"][:]
	Semimajor_Axis_Direction = data_file["Semimajor_Axis_Direction"][:]
	q = data_file["q"][:]
	data_file.close()
	data_dir = {
		"Position": COM,
		"Position_shape_sample": COM,
		"Axis_Direction": Semimajor_Axis_Direction,
		"LOS": 2,
		"q": q
	}
	return MeasureIABox(data_dir, f"./data/processed/TNG300/test_IA_mock_TNG300_large.hdf5",
					 "TNG300", 99, [0.1, 20], 10, 8,
						None, num_nodes=8)


@pytest.fixture()
def IA_mock_TNG300_n17_large():
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300_large.hdf5', 'a')
	COM = data_file["COM"][:]
	Semimajor_Axis_Direction = data_file["Semimajor_Axis_Direction"][:]
	q = data_file["q"][:]
	data_file.close()
	data_dir = {
		"Position": COM,
		"Position_shape_sample": COM,
		"Axis_Direction": Semimajor_Axis_Direction,
		"LOS": 2,
		"q": q
	}
	return MeasureIABox(data_dir, f"./data/processed/TNG300/test_IA_mock_TNG300_large.hdf5",
					 "TNG300", 99, [0.1, 20], 10,
						8, None, num_nodes=17)


@pytest.fixture()
def IA_available_output():
	return MeasureIABox(None, f"./data/processed/TNG300/mock_IA_TNG300.hdf5",
						"TNG300", 99, [0.1, 20], 10,
						8)


@pytest.fixture()
def jk_regions():
	COM = np.array([[1, 1, 1], [2, 1, 2], [2.5, 2.5, 1.51], [1, 2, 2]])
	data_dict = {
		"Position": COM,
		"Position_shape_sample": COM,
		"Axis_Direction": np.array([]),
		"LOS": 2,
		"q": np.array([]),
	}
	return MeasureIABox(data_dict, f"./data/processed/TNG300/no_write.hdf5", None, None, boxsize=3.)


@pytest.fixture()
def IA_mock_TNG300_jk():
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300.hdf5', 'a')
	COM = data_file["COM"][:]
	Semimajor_Axis_Direction = data_file["Semimajor_Axis_Direction"][:]
	q = data_file["q"][:]
	data_file.close()
	data_dir = {
		"Position": COM,
		"Position_shape_sample": COM,
		"Axis_Direction": Semimajor_Axis_Direction,
		"LOS": 2,
		"q": q
	}
	return MeasureIABox(data_dir, f"./data/processed/TNG300/test_IA_mock_TNG300_jk.hdf5",
						"TNG300", 99, [0.1, 20], 10,
						8, None)


@pytest.fixture()
def IA_mock_TNG300_jk_n8():
	data_file = h5py.File('./data/raw/TNG300/mock_TNG300.hdf5', 'a')
	COM = data_file["COM"][:]
	Semimajor_Axis_Direction = data_file["Semimajor_Axis_Direction"][:]
	q = data_file["q"][:]
	data_file.close()
	data_dir = {
		"Position": COM,
		"Position_shape_sample": COM,
		"Axis_Direction": Semimajor_Axis_Direction,
		"LOS": 2,
		"q": q
	}
	return MeasureIABox(data_dir, f"./data/processed/TNG300/test_IA_mock_TNG300_jk.hdf5",
						"TNG300", 99, [0.1, 20], 10,
						8, None, num_nodes=8)
