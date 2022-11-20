import pandas as pd
from nlp_utils import *
from cleaning_pipeline import *
import time
from packages import *
from bert import BertClassifier
from tfidf import TfidfClassifer
from stacking import Stacking
from sentiment import Sentiment
import warnings

warnings.filterwarnings("ignore")


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
            stopwords_bool=True,
            lemmatize_bool=True,
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
        self.results_path = self.out_path + self.timestamp + "_CMACGM_Results" + ".csv"

    def initalize_model(self):

        NLPClassifier = BertClassifier(self.data, self.local_path, cuda=False)
        NLPClassifier.predict()

        print("\n Running TFIDF classification... \n")
        EmailObjectClassifier = TfidfClassifer(
            self.data, self.local_path, "EmailObject", train=False
        )
        EmailObjectClassifier.predict()

        LastEmailContentClassifier = TfidfClassifer(
            self.data, self.local_path, "LastEmailContent", train=False
        )
        LastEmailContentClassifier.predict()

        TeamNameClassifier = TfidfClassifer(
            self.data, self.local_path, "TeamName", train=False
        )
        TeamNameClassifier.predict()

        ContactEmailClassifier = TfidfClassifer(
            self.data, self.local_path, "ContactEmail", train=False
        )
        ContactEmailClassifier.predict()

        print("\nTFIDF classification done. \n")

        NLP_Sentiment = Sentiment(self.data, self.local_path)
        NLP_Sentiment.predict()

        # Get predictions
        bert_classifier_preds = NLPClassifier.predictions
        email_object_preds = EmailObjectClassifier.predictions
        email_content_preds = LastEmailContentClassifier.predictions
        team_name_preds = TeamNameClassifier.predictions
        contact_email_preds = ContactEmailClassifier.predictions
        sentiment_preds_labels = NLP_Sentiment.predictions_labels
        sentiment_preds_score = NLP_Sentiment.predictions_score

        d = {
            "bert_classifier_preds": bert_classifier_preds,
            "email_object_preds": email_object_preds,
            "email_content_preds": email_content_preds,
            "team_name_preds": team_name_preds,
            "contact_email_preds": contact_email_preds,
            "sentiment_preds_labels": sentiment_preds_labels,
            "sentiment_preds_score": sentiment_preds_score,
        }

        self.ext_preds = pd.DataFrame(d)
        Stacker = Stacking(self.data, self.local_path, self.ext_preds)
        self.data["PredictedType"] = Stacker.predict()
        self.data.to_csv(self.results_path, index=False)

        print("\n\n Model finished running. \n Results saved in: ", self.results_path)
