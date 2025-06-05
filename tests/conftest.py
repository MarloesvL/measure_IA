import pytest
import h5py
from src.measure_SV import MeasureSnapshotVariables
from src.measure_IA_base import MeasureIABase
from src.measure_IA import MeasureIA


# this file has to be called conftest otherwise the fixtures will not work

# example
# @pytest.fixture()
# def sim():
# 	return demo_sim(4)
#
# def demo_sim(a):
# 	return a**2

@pytest.fixture()
def SV_PT4_TNG100_99():
	raw_path = "./data/raw/"
	return MeasureSnapshotVariables(4, "TNG100", 99, numnodes=1, output_file_name=None, data_path=raw_path, update=True)


@pytest.fixture()
def SV_PT4_TNG300_99():
	raw_path = "./data/raw/"
	return MeasureSnapshotVariables(4, "TNG300", 99, numnodes=1, output_file_name=None, data_path=raw_path, update=True)


@pytest.fixture()
def SV_PT4_EAGLE_28():
	raw_path = "./data/raw/"
	return MeasureSnapshotVariables(4, "EAGLE", 28, numnodes=1, output_file_name=None, data_path=raw_path, update=True)


@pytest.fixture()
def IA_TNG100_99():
	data = {}
	return MeasureIABase(data, simulation="TNG100", snapshot=99, separation_limits=[], num_bins_r=1, num_bins_pi=2)


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
	return MeasureIA(data_dir, "TNG300", 99, [0.1, 20], 10, 8, 4, None,
					 f"./data/processed/TNG300/test_IA_mock_TNG300.hdf5",
					 num_nodes=8)


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
	return MeasureIA(data_dir, "TNG300", 99, [0.1, 20], 10, 8, 4, None,
					 f"./data/processed/TNG300/test_IA_mock_TNG300_large.hdf5",
					 num_nodes=1)


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
	return MeasureIA(data_dir, "TNG300", 99, [0.1, 20], 10, 8, 4, None,
					 f"./data/processed/TNG300/test_IA_mock_TNG300_large.hdf5",
					 num_nodes=8)


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
	return MeasureIA(data_dir, "TNG300", 99, [0.1, 20], 10, 8, 4, None,
					 f"./data/processed/TNG300/test_IA_mock_TNG300_large.hdf5",
					 num_nodes=17)

# def MSV(project, snapshot):
# 	raw_path = "./data/raw/"
# 	SV_obj = MeasureSnapshotVariables(project, snapshot, numnodes=1, output_file_name=None,
# 									  data_path=raw_path,
# 									  update=True)
# 	SV_obj.get_folders(fof_folder="", snap_folder="")
# 	return SV_obj
