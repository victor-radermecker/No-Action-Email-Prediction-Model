import pandas as pd
from nlp_utils import *
from cleaning_pipeline import *
import time
from packages import *
from bert import BertClassifier
from tfidf import TfidfClassifer

class EmailClassifier:
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path

        # Data PreProcessing / Cleaning
        self.set_output_paths()
        self.read_data()
        self.clean_data()

        # get local path
        self.local_path = os.path.dirname(os.path.abspath(__file__))

        # Model
        self.initalize_model()

    def init(self):
        pass

    def info(self):
        pass

    def read_data(self):
        """
        Read data from a csv file.
        :param path: path to the csv file
        :return: pandas dataframe

        (!) The structure of the CSV File should follow the instructions of the README.md file. (!)
        """
        self.data = pd.read_csv(self.in_path)

    def clean_data(self):
        """
        Clean the data using the cleaning pipeline.
        :return: None
        """
        self.data = data_cleaning(
            self.data,
            self.cleaned_data_path,
            stopwords_bool=False,
            lemmatize_bool=False,
            english_words_bool=False,
        )

    def set_output_paths(self):
        """
        Set the paths to the output files.
        :return: None
        """
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.cleaned_data_path = (
            self.out_path + self.timestamp + "_CMACGM_DataFrame_Clean" + ".csv"
        )
        self.results_path = (
            self.out_path + "self.timestamp" + "_CMACGM_Results" + ".csv"
        )

    def initalize_model(self):
        # BertClassifier(self.data, self.local_path, cuda=False)
        EmailObjectClassifier = TfidfClassifer(self.data, self.local_path, "EmailObject", train=False)
        EmailObjectClassifier.predict()
        print(EmailObjectClassifier.predictions)


