import os 

DATA_PATH = os.path.join('..', 'data', 'IDA 2024 Industrial Challenge')
TRAIN_OPERATIONAL_PATH = os.path.join(DATA_PATH, 'train_operational_readouts.csv')
TRAIN_REPAIR_PATH = os.path.join(DATA_PATH, 'train_tte.csv')
TRAIN_SPECIFICATIONS = os.path.join(DATA_PATH, 'train_specifications.csv')

TEST_PATH = os.path.join(DATA_PATH, 'public_X_test.csv')
VARIANTS_PATH = os.path.join(DATA_PATH, 'variants.csv')