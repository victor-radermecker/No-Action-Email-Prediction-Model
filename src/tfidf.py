from packages import *
import joblib


class TfidfClassifer:
    def __init__(self, data, local_path, feature, train=False):
        self.data = data.copy(deep=True)
        self.local_path = local_path
        self.feature = feature

        # Start testing
        self.preprocess_data()
        self.select_features()

    def preprocess_data(self):
        # Remove stopwords
        lst_stopwords = nltk.corpus.stopwords.words("english")
        self.data[self.feature] = self.data[self.feature].progress_apply(
            lambda x: self.utils_preprocess_text(
                x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords
            )
        )

    def select_features(self):

        if self.feature == "EmailObject":
            self.corpus = self.data["EmailObject"]
        elif self.feature == "LastEmailContent":
            self.corpus = self.data["LastEmailContent"]
        elif self.feature == "TeamName":
            self.data["TeamName"] = self.data["TeamName"].str.replace("-", " ")
            self.corpus = self.data["TeamName"]
        elif self.feature == "ContactEmail":
            self.data["ContactEmail"] = self.data["ContactEmail"].str.replace("@", " ")
            self.data["ContactEmail"] = self.data["ContactEmail"].str.replace(".", " ")
            self.data["ContactEmail"] = self.data["ContactEmail"].str.replace("_", " ")
            self.data["ContactEmail"] = self.data["ContactEmail"].str.replace("-", " ")
            self.data["LastEmailCCAddress"] = self.data[
                "LastEmailCCAddress"
            ].str.replace("@", " ")
            self.data["LastEmailCCAddress"] = self.data[
                "LastEmailCCAddress"
            ].str.replace(".", " ")
            self.data["LastEmailCCAddress"] = self.data[
                "LastEmailCCAddress"
            ].str.replace("_", " ")
            self.data["LastEmailCCAddress"] = self.data[
                "LastEmailCCAddress"
            ].str.replace("-", " ")
            self.data["Emails"] = (
                self.data["ContactEmail"] + self.data["LastEmailCCAddress"]
            )
            self.corpus = self.data["Emails"]
        else:
            print("Error: Feature not found")

    def train(self):

        # vectorize Data
        self.df_train, self.df_test = train_test_split(
            self.data, test_size=0.1, random_state=1, stratify=self.data["Type"]
        )
        self.vectorizer, self.X_train = self.initVectorizer(self.df_train, self.feature)

        ## get target
        self.y_train = self.df_train["Type"].values
        self.y_test = self.df_test["Type"].values

        # pipeline
        classifier = RandomForestClassifier(verbose=0)
        model_rf = pipeline.Pipeline(
            [("vectorizer", self.vectorizer), ("classifier", classifier)]
        )

        ## train classifier
        model_rf["classifier"].fit(self.X_train, self.y_train)

        # save model
        model_dir = (
            self.local_path
            + "/../"
            + "models/tfidf/"
            + time.strftime("%Y%m%d-%H%M%S_")
            + self.feature
            + "Classifier.joblib"
        )
        pickle.dump(model_rf, open(model_dir, "wb"))

    def predict(self):
        model_dir = (
            self.local_path
            + "/../"
            + "models/tfidf/"
            + self.feature
            + "Classifier.joblib"
        )
        model = pickle.load(open(model_dir, "rb"))
        self.predictions = model.predict(self.data[self.feature].values)

    def utils_preprocess_text(
        self, text, flg_stemm=False, flg_lemm=True, lst_stopwords=None
    ):
        """
        Preprocess a string.
        :parameter
            :param text: string - name of column containing text
            :param lst_stopwords: list - list of stopwords to remove
            :param flg_stemm: bool - whether stemming is to be applied
            :param flg_lemm: bool - whether lemmitisation is to be applied
        :return
            cleaned text
        """
        ## clean (convert to lowercase and remove punctuations and characters and then strip)
        text = re.sub(r"[^\w\s]", "", str(text).lower().strip())

        ## Tokenize (convert from string to list)
        lst_text = text.split()
        ## remove Stopwords
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in lst_stopwords]

        ## Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]

        ## Lemmatisation (convert the word into root word)
        if flg_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]

        ## back to string from list
        text = " ".join(lst_text)
        return text

    def initVectorizer(self):
        """
        Initialize the vectorizer.
        """

        ## Tf-Idf (advanced variant of BoW)
        vectorizer = feature_extraction.text.TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2)
        )

        vectorizer.fit(self.corpus)
        X_train = vectorizer.transform(self.corpus)
        y = self.data["Type"]

        X_names = vectorizer.get_feature_names()
        p_value_limit = 0.95
        dtf_features = pd.DataFrame()

        for cat in np.unique(y):
            chi2, p = feature_selection.chi2(X_train, y == cat)
            dtf_features = dtf_features.append(
                pd.DataFrame({"feature": X_names, "score": 1 - p, "y": cat})
            )

            dtf_features = dtf_features.sort_values(
                ["y", "score"], ascending=[True, False]
            )

            dtf_features = dtf_features[dtf_features["score"] > p_value_limit]

        X_names = dtf_features["feature"].unique().tolist()

        for cat in np.unique(y):
            print("# {}:".format(cat))
            print("  . selected features:", len(dtf_features[dtf_features["y"] == cat]))
            print(
                "  . top features:",
                ",".join(dtf_features[dtf_features["y"] == cat]["feature"].values[:10]),
            )
            print(" ")

        self.vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
        self.vectorizer.fit(self.corpus)
        self.X_train = self.vectorizer.transform(self.corpus)
