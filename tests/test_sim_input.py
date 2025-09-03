from src.measure_IA import MeasureIA


def test_sim_input():
	sim_str = "TNG300"
	boxsize = 300.
	snapshot = 30

	siminfo_v = MeasureIA(None, sim_str, snapshot)
	siminfo_nosnap_v = MeasureIA(None, sim_str, None)
	boxsize_v = MeasureIA(None, None, snapshot, boxsize=boxsize)
	boxsize_nosnap_v = MeasureIA(None, None, None, boxsize=boxsize)
	nosim_v = MeasureIA(None, LOS_lim=20.)
	siminfo_extrabox_v = MeasureIA(None, sim_str, snapshot, boxsize=boxsize)

	assert siminfo_v.simname == sim_str
	assert siminfo_v.snapshot == '30'
	assert siminfo_v.boxsize == 205.
	assert siminfo_v.snap_group == "Snapshot_30/"

	assert siminfo_nosnap_v.simname == sim_str
	assert siminfo_nosnap_v.snapshot is None
	assert siminfo_nosnap_v.boxsize == 205.
	assert siminfo_nosnap_v.snap_group == ""

	assert boxsize_v.simname is None
	assert boxsize_v.snapshot == '30'
	assert boxsize_v.boxsize == 300.
	assert boxsize_v.snap_group == "Snapshot_30/"

	assert boxsize_nosnap_v.simname is None
	assert boxsize_nosnap_v.snapshot is None
	assert boxsize_nosnap_v.boxsize == 300.
	assert boxsize_nosnap_v.snap_group == ""

	assert nosim_v.simname is None
	assert nosim_v.snapshot is None
	assert nosim_v.boxsize is None
	assert nosim_v.snap_group == ""

	assert siminfo_extrabox_v.simname == sim_str
	assert siminfo_extrabox_v.snapshot == '30'
	assert siminfo_extrabox_v.boxsize == 205.
	assert siminfo_extrabox_v.snap_group == "Snapshot_30/"

	return


if __name__ == '__main__':
	test_sim_input()
