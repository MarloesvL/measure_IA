# import internal classes for use, so the module file names do not need to be called.

from importlib.metadata import PackageNotFoundError, version as _version

try:
	# read from the installed distribution metadata, so the version can never drift
	# from the one declared in pyproject.toml
	__version__ = _version("measureia")
except PackageNotFoundError:  # running from a source tree that was never installed
	__version__ = "unknown"

# import base and wrapper classes
from .measure_IA import MeasureIABox
from .measure_IA_lightcone import MeasureIALightcone
from .measure_IA_base import MeasureIABase
from .check_input import CheckInput

# import covariance measurement class
from .measure_jackknife import MeasureJackknife

# import backend method classes used in MeasureIA
from .measure_w_box import MeasureWBox
from .measure_m_box import MeasureMultipolesBox
from .measure_w_box_jk import MeasureWBoxJackknife
from .measure_m_box_jk import MeasureMBoxJackknife
from .measure_galaxy_box import MeasureGalaxyContributionsBox
from .measure_w_lightcone import MeasureWLightcone
from .measure_m_lightcone import MeasureMultipolesLightcone
from .measure_w_lightcone_jk import MeasureWLightconeJackknife
from .measure_m_lightcone_jk import MeasureMultipolesLightconeJackknife

# import utilities
from .read_data import ReadData
from .Sim_info import SimInfo
from .write_data import create_group_hdf5, write_dataset_hdf5
