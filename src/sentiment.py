from packages import *
import transformers
from transformers.pipelines.pt_utils import KeyDataset


# from transformers import pipeline
# from transformers import AutoTokenizer


class Sentiment:
    def __init__(self, data, local_path):
        self.data = data.copy(deep=True)
        self.local_path = local_path

        # Cleaning pipeline
        self.clean()

    def clean(self):
        self.data["LastEmailContent"] = self.data["LastEmailContent"].apply(
            lambda x: self.clean_up_pipeline(x)
        )

    def make_str(self, word):
        if not isinstance(word, str):
            return ""
        return str(word)

    def introduce_empty(self, word):
        if not word or not isinstance(word, str) or len(word) == 0:
            return "[EMPTY]"
        return word

    def clean_up_pipeline(self, sentence):
        cleaning_utils = [
            self.make_str,
            # remove_hyperlink,
            # replace_newline,
            # to_lower,
            # remove_number,
            # remove_punctuation,
            # remove_whitespace,
            self.introduce_empty,
        ]

        for o in cleaning_utils:
            sentence = o(sentence)
        return sentence

    def predict(self):

        print("\n Running sentiment classification...")

        tokenizer = "distilbert-base-uncased-finetuned-sst-2-english"
        model = "distilbert-base-uncased-finetuned-sst-2-english"

        sentClassifier = transformers.pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )

        self.datalist = self.data["LastEmailContent"].tolist()

        self.predictions_labels = []
        self.predictions_score = []

        for i in tqdm(range(len(self.datalist))):
            prediction = sentClassifier(
                [self.datalist[i]], padding=True, truncation=True
            )
            self.predictions_labels.append(prediction[0]["label"])
            self.predictions_score.append(prediction[0]["score"])

        # map negative sentiment to 0 and positive to 1

        # change the label to 0 if negative and 1 if positive
        self.predictions_labels = [
            0 if x == "NEGATIVE" else 1 for x in self.predictions_labels
        ]

        print("\n Sentiment classification done.")
