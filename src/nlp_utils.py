# import re
# from tqdm import tqdm

# tqdm.pandas()
# import nltk

# import pandas as pd
# from tqdm import tqdm

# tqdm.pandas()
# import numpy as np
# import nltk
# from sklearn import feature_extraction, feature_selection

# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")


# def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
#     """
#     Preprocess a string.
#     :parameter
#         :param text: string - name of column containing text
#         :param lst_stopwords: list - list of stopwords to remove
#         :param flg_stemm: bool - whether stemming is to be applied
#         :param flg_lemm: bool - whether lemmitisation is to be applied
#     :return
#         cleaned text
#     """
#     ## clean (convert to lowercase and remove punctuations and characters and then strip)
#     text = re.sub(r"[^\w\s]", "", str(text).lower().strip())

#     ## Tokenize (convert from string to list)
#     lst_text = text.split()
#     ## remove Stopwords
#     if lst_stopwords is not None:
#         lst_text = [word for word in lst_text if word not in lst_stopwords]

#     ## Stemming (remove -ing, -ly, ...)
#     if flg_stemm == True:
#         ps = nltk.stem.porter.PorterStemmer()
#         lst_text = [ps.stem(word) for word in lst_text]

#     ## Lemmatisation (convert the word into root word)
#     if flg_lemm == True:
#         lem = nltk.stem.wordnet.WordNetLemmatizer()
#         lst_text = [lem.lemmatize(word) for word in lst_text]

#     ## back to string from list
#     text = " ".join(lst_text)
#     return text


# # Instantiate Vectorizer
# def initVectorizer(data, var_used):

#     ## Tf-Idf (advanced variant of BoW)
#     vectorizer = feature_extraction.text.TfidfVectorizer(
#         max_features=10000, ngram_range=(1, 2)
#     )

#     corpus = data[var_used]

#     vectorizer.fit(corpus)
#     X_train = vectorizer.transform(corpus)
#     dic_vocabulary = vectorizer.vocabulary_

#     y = data["Type"]

#     X_names = vectorizer.get_feature_names()
#     p_value_limit = 0.95
#     dtf_features = pd.DataFrame()

#     for cat in np.unique(y):
#         chi2, p = feature_selection.chi2(X_train, y == cat)
#         dtf_features = dtf_features.append(
#             pd.DataFrame({"feature": X_names, "score": 1 - p, "y": cat})
#         )

#         dtf_features = dtf_features.sort_values(["y", "score"], ascending=[True, False])

#         dtf_features = dtf_features[dtf_features["score"] > p_value_limit]

#     X_names = dtf_features["feature"].unique().tolist()

#     for cat in np.unique(y):
#         print("# {}:".format(cat))
#         print("  . selected features:", len(dtf_features[dtf_features["y"] == cat]))
#         print(
#             "  . top features:",
#             ",".join(dtf_features[dtf_features["y"] == cat]["feature"].values[:10]),
#         )
#         print(" ")

#     vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
#     vectorizer.fit(corpus)
#     X_train = vectorizer.transform(corpus)
#     dic_vocabulary = vectorizer.vocabulary_

#     return vectorizer, X_train
