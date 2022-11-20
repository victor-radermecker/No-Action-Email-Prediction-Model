### TO CLEAN

import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import os
import time
import datetime
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm

tqdm.pandas()
import nltk

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import numpy as np
import nltk
from sklearn import feature_extraction, feature_selection

# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")


# remove warnings
import warnings

warnings.filterwarnings("ignore")

## for data
import json
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import re
import nltk

## for bag-of-words
from sklearn import (
    feature_extraction,
    model_selection,
    naive_bayes,
    pipeline,
    manifold,
    preprocessing,
    feature_selection,
)
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## for explainer
from lime import lime_text

## for word embedding
import gensim
import gensim.downloader as gensim_api
from sklearn import metrics

# import pickle
import joblib
import pickle

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import xgboost as xgb
from xgboost import XGBClassifier
from tqdm import tqdm
