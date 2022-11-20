from packages import *


class Sentiment:
    def __init__(self, data, local_path):
        self.data = data
        self.local_path = local_path

        # Cleaning pipeline
        self.clean()

    def clean(self):
        self.data["LastEmailContent"] = self.data["LastEmailContent"].apply(
            lambda x: self.clean_up_pipeline(x)
        )

    def predict(self):
        pass

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
