# ©, 2022, Sirris
# owner: FFNG
""" All relevant paths are stored in the __init__.py file:

- locally saved files and models: LOCAL_PATH_DATA
- remotely saved unprocessed data files: REMOTE_DATA_PATH
- remotely saved processed data files and models: REMOTE_OUTPUTS_PATH
- figures are only saved locally: BASE_PATH_FIGURES
"""

import os
import posixpath
from pathlib import Path
import pkgutil as _pkgutil

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# remote data paths
REMOTE_DATA_PATH = posixpath.join('data', 'CONSCIOUS', 'ENGIE-I-care')
REMOTE_DATA_PATH_OPERATIONAL_ZIP_1 = posixpath.join(REMOTE_DATA_PATH, 'operational_data_211117.zip')
REMOTE_DATA_PATH_OPERATIONAL_ZIP_2 = posixpath.join(REMOTE_DATA_PATH, 'operational_data_241122.zip')

# remote output paths
REMOTE_OUTPUTS_PATH = posixpath.join('outputs', 'CONSCIOUS', 'Engie')
REMOTE_OUTPUTS_PATH_RAW_OPERATIONAL_DATA_1 = posixpath.join(REMOTE_OUTPUTS_PATH, 'operational_data_211117')  # !! still remote_path_operational
REMOTE_OUTPUTS_PATH_RAW_OPERATIONAL_DATA_2 = posixpath.join(REMOTE_OUTPUTS_PATH, 'operational_data_221124')
REMOTE_OUTPUTS_PATH_VIBRATION_1 = posixpath.join(REMOTE_OUTPUTS_PATH, 'vibration_data_211117')
REMOTE_OUTPUTS_PATH_VIBRATION_1_BINNED = posixpath.join(REMOTE_OUTPUTS_PATH_VIBRATION_1, 'processed')
# TODO: remote path vibration 2
REMOTE_OUTPUTS_PATH_VIBRATION_2 = posixpath.join(REMOTE_OUTPUTS_PATH, 'vibration_data_220912')
REMOTE_OUTPUTS_PATH_MODELS = posixpath.join(REMOTE_OUTPUTS_PATH, 'models')
REMOTE_OUTPUTS_PATH_NMF_MODELS = posixpath.join(REMOTE_OUTPUTS_PATH_MODELS, 'nmf_vibration_profiling')

# local paths
LOCAL_PATH_DATA = os.path.join(ROOT_DIR, 'data')
LOCAL_TEST_PATH = os.path.join(LOCAL_PATH_DATA, 'test')
LOCAL_PATH_DATA_ZIPPED_OPERATIONAL_1 = os.path.join(LOCAL_PATH_DATA, 'operational_data_211117.zip')
LOCAL_PATH_DATA_RAW_OPERATIONAL_1 = os.path.join(LOCAL_PATH_DATA, 'operational_data_211117')
LOCAL_PATH_DATA_ZIPPED_OPERATIONAL_2 = os.path.join(LOCAL_PATH_DATA, 'operational_data_221124.zip')
LOCAL_PATH_DATA_RAW_OPERATIONAL_2 = os.path.join(LOCAL_PATH_DATA, 'operational_data_221124')
LOCAL_PATH_VIBRATION_1 = os.path.join(LOCAL_PATH_DATA, 'vibration_data_211117', 'preprocessed')
LOCAL_PATH_VIBRATION_2 = os.path.join(LOCAL_PATH_DATA, 'vibration_data_220912', 'preprocessed')
LOCAL_PATH_DATA_RAW_OPERATIONAL = os.path.join(LOCAL_PATH_DATA, 'operational_data')  # the merged operational files
LOCAL_PATH_MODELS = os.path.join(LOCAL_PATH_DATA, 'models')
LOCAL_PATH_NMF_MODELS = os.path.join(LOCAL_PATH_MODELS, 'nmf_vibration_profiling')

# set local base path
# previously with old data: LOCAL_PATH_OPERATIONAL = os.path.join('..', 'data', 'operational_data_211117')
LOCAL_PATH_OPERATIONAL = LOCAL_PATH_DATA_RAW_OPERATIONAL
#LOCAL_PATH_VIBRATION = os.path.join('..', 'data', 'vibration_data_211117', 'preprocessed')

# set remote outputs base path
REMOTE_PATH_OPERATIONAL = posixpath.join(REMOTE_OUTPUTS_PATH, 'operational_data')
#REMOTE_PATH_VIBRATION = posixpath.join(REMOTE_PATH_ENGIE, 'vibration_data_211117')

# if the clustering changes, please adapt the path below
LATEST_EXPERIMENT = '27-02-23'
LAST_OPERATIONAL_MODEL_FOLDER = os.path.join(LOCAL_PATH_OPERATIONAL, LATEST_EXPERIMENT)
LATEST_CLUSTERING_NAME = f'clustering_{LATEST_EXPERIMENT}.csv'
LOCAL_PATH_OPERATIONAL_DATAFRAME = os.path.join(LAST_OPERATIONAL_MODEL_FOLDER, LATEST_CLUSTERING_NAME)
REMOTE_OUTPUTS_PATH_OPERATIONAL_DATAFRAME = posixpath.join(REMOTE_PATH_OPERATIONAL, LATEST_CLUSTERING_NAME)

BASE_PATH_FIGURES = os.path.join(ROOT_DIR, 'work', 'figs')
