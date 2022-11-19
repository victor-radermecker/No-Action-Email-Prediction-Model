import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# identify digits from text
from string import digits

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('words')

# detect language from text
import langid

tqdm.pandas()


def data_cleaning(df, path, stopwords=True, lemmatize=True, english_words=True):
    """
    df: raw dataframe from CMA CGM.
    """

    print("Cleaning starting...")

    # Drop columns with no email content
    df = df.dropna(subset=["LastIncomingEmail__c"])

    # Drop this column as it contains always the same string "Case" or "Contact"
    df = df.drop(["attributes.type", "Contact.attributes.type"], axis=1)

    # drop the columns with only null values
    df = df.drop(["Contact"], axis=1)

    # create a new column for the email content initalized to empty string
    df["LastIncomingEmailContent"] = ""

    # convert to TYPE column to 1 if NOAC and to 0 otherwise
    df["Type"] = df["Type"].apply(lambda x: 1 if x == "NOAC" else 0)

    # assign the number of times ; appears in each string of LastEmailCCAddress__c in the new column cc_count
    df["LastEmailCCAddressCount"] = df["LastEmailCCAddress__c"].str.count(";") + 1

    # fill NaN of LastEmailCCAddress__count by 0
    df["LastEmailCCAddressCount"] = df["LastEmailCCAddressCount"].fillna(0)
    df["LastEmailCCAddressCount"] = df["LastEmailCCAddressCount"].astype(int)

    # check when SuppliedEmail has the same values as ContactEmail
    test = df.apply(
        lambda row: 1 if row["SuppliedEmail"] == row["Contact.Email"] else 0, axis=1
    )

    # Remove the CaseNumber from the EmailTemplateSubjectDispute__c
    df["EmailTemplateSubjectDispute__c"] = df.apply(
        lambda row: row["EmailTemplateSubjectDispute__c"].replace(
            "Case #" + str(row["CaseNumber"]), ""
        ),
        axis=1,
    )

    # convert Contact.attributes.url to string
    df["Contact.attributes.url"] = df["Contact.attributes.url"].astype(str)

    # remove /services/data/v42.0/sobjects/Contact/ from the strings in Contact.attributes.url
    df["Contact.attributes.url"] = df["Contact.attributes.url"].apply(
        lambda x: x.replace("/services/data/v42.0/sobjects/Contact/", "")
    )

    # convert Contact.attributes.url to string
    df["attributes.url"] = df["attributes.url"].astype(str)

    # remove /services/data/v42.0/sobjects/Contact/ from the strings in Contact.attributes.url
    df["attributes.url"] = df["attributes.url"].apply(
        lambda x: x.replace("/services/data/v42.0/sobjects/Case/", "")
    )

    # For the vast majority of the rows (85%), the columns SuppliedEmail and Contact.Email have the same value.
    # When the values are different, one of them has NaN and the other has the email address.
    # Therefore, we can fill the NaN of Contact.Email with the value of SuppliedEmail
    df["Contact.Email"] = df["Contact.Email"].fillna(df["SuppliedEmail"])

    # Then we can drop the column SuppliedEmail
    df = df.drop(["SuppliedEmail"], axis=1)

    # fill NaN of LastEmailCCAddress__c by empty string
    df["LastEmailCCAddress__c"] = df["LastEmailCCAddress__c"].fillna("")

    # Rename Topics__c in Topics
    df = df.rename(columns={"Topics__c": "Topics"})

    # Rename LastIncomingEmail__c in LastIncomingEmail
    df = df.rename(columns={"LastIncomingEmail__c": "LastIncomingEmail"})

    # Rename TeamName__c in TeamName
    df = df.rename(columns={"TeamName__c": "TeamName"})

    # Rename RequesterEmail__c in RequesterEmail
    df = df.rename(columns={"RequesterEmail__c": "RequesterEmail"})

    # Rename EmailTemplateSubjectDispute__c by EmailObject
    df = df.rename(columns={"EmailTemplateSubjectDispute__c": "EmailObject"})

    # Rename LastEmailCCAddress__c by LastEmailCCAddress
    df = df.rename(columns={"LastEmailCCAddress__c": "LastEmailCCAddress"})

    # Rename attributes.url by AttributesURL
    df = df.rename(columns={"attributes.url": "AttributesURL"})

    # Rename Contact.attributes.url by ContactAttributesURL
    df = df.rename(columns={"Contact.attributes.url": "ContactAttributesURL"})

    # Rename Contact.Email by ContactEmail
    df = df.rename(columns={"Contact.Email": "ContactEmail"})

    # Cleaning of the LastIncomingEmail column using BeautifulSoup
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        df.at[index, "LastIncomingEmailContent"] = BeautifulSoup(
            row["LastIncomingEmail"], "html"
        ).text

    # drop the column LastIncomingEmail
    df = df.drop(["LastIncomingEmail"], axis=1)

    # reset index
    df.reset_index(inplace=True)

    # From LastEmailCCAddress get whether cma-cgm is cc'd or not
    df["CMA_in_cc"] = df["LastEmailCCAddress"].str.find("cma-cgm") > -1

    # new feature: how many emails are exchanged in each conversation, counting Sent:
    df["CountMailsInConversation"] = [
        len(re.findall("Sent:", str(df["LastIncomingEmailContent"][i]))) + 1
        for i in range(len(df["LastIncomingEmailContent"]))
    ]

    df = df.replace(r"\n\n", " ", regex=True)
    df = df.replace(r"  ", " ", regex=True)
    df = df.replace(r"\t\t", " ", regex=True)

    df["LastEmailContent"] = df["LastIncomingEmailContent"].progress_apply(split_emails)

    # save the cleaned data
    df.to_csv(
        path,
        index=False,
    )

    print("Cleaning completed. :) ")

    return df


def parenthesis_cleaner(test_str):
    ret = ""
    skip1c = 0
    skip2c = 0
    for i in test_str:
        if i == "{":
            skip1c += 1
        elif i == "[":
            skip2c += 1
        elif i == "]" and skip1c > 0:
            skip1c -= 1
        elif i == "}" and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret


def split_emails(text, stopwords, lemmatize, english_words):

    # convert text to string
    text = str(text)
    text = parenthesis_cleaner(text)

    if text.find("From:"):
        # cut the text after the first occurrence of "From:"
        text = text.split("From:")[0]
    elif text.find("Sent:"):
        # cut the text after the first occurrence of "Sent:"
        text = text.split("Sent:")[0]
    elif text.find("To:"):
        # cut the text after the first occurrence of "To:"
        text = text.split("To:")[0]
    elif len(text) < 25:
        text = " "
    else:
        text = text

    # Clean the rest of the text

    # remove html tags
    text = re.sub(r"<.*?>", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove all numbers from emails, they are not relevant
    text = text.translate(str.maketrans("", "", digits))

    if stopwords:
        # remove stopwords
        text_tokens = word_tokenize(text)
        text = " ".join([word for word in text_tokens if not word in stopwords.words()])

    if lemmatize:
        # define lemmatizer
        lemmatizer = WordNetLemmatizer()

        # word lemmatization
        lemma_words = [lemmatizer.lemmatize(o) for o in text.split()]
        text = " ".join(lemma_words)

    if english_words:
        # remove non english words from string (advanced, takes a lot of time)
        words = set(nltk.corpus.words.words())
        text = " ".join(
            w
            for w in nltk.wordpunct_tokenize(text)
            if w.lower() in words or not w.isalpha()
        )

    return text
