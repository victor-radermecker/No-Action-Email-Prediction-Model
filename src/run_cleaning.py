import pandas as pd
from cleaning_pipeline import *

import warnings

warnings.filterwarnings("ignore")

# Store data locally for confidentiality reasons!
data_path = "/Users/victor/Documents/Confidential Dataset/ML_NOAC_NOVA_Extraction.csv"

# Read raw data
df = pd.read_csv(data_path)
data_cleaning(df)
