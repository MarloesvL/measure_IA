import pytest
from src.measure_SV import MeasureSnapshotVariables


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

# def MSV(project, snapshot):
# 	raw_path = "./data/raw/"
# 	SV_obj = MeasureSnapshotVariables(project, snapshot, numnodes=1, output_file_name=None,
# 									  data_path=raw_path,
# 									  update=True)
# 	SV_obj.get_folders(fof_folder="", snap_folder="")
# 	return SV_obj
