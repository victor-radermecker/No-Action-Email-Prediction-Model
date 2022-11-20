from EmailClassifier import EmailClassifier

import sys
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    print("Welcome to the email classifier of CMA CGM!")

    input_path = sys.argv[1] # where you saved the data you want to classify
    output_path = sys.argv[2] # where you want to save the results

    model = EmailClassifier(input_path, output_path)
    # This main should take a path to a csv file as input and output a number (classification).
