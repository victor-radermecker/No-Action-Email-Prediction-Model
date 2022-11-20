from packages import *


class Stacking:
    def __init__(self, data, local_path, ext_preds):
        self.data = data.copy(deep=True)
        self.local_path = local_path
        self.ext_preds = ext_preds

        # Preprocess
        self.preprocess()

    def train(self):
        pass  # TODO

    def predict(self):
        model_dir = self.local_path + "/../" + "models/stacking/" + "StackingXGBoost"
        # model = pickle.load(open(model_dir, "rb"))

        bst = XGBClassifier()  # init model
        bst.load_model(model_dir)  # load data

        if "Type" in self.data.columns:
            self.data.drop("Type", axis=1, inplace=True)

        return bst.predict(self.data)

    def preprocess(self):
        # create features with fourth and last word in "TeamName"
        self.data["TeamName_division"] = self.data["TeamName"].apply(
            lambda x: x.split("-")[3]
        )
        self.data["TeamName_client"] = self.data["TeamName"].apply(
            lambda x: x.split("-")[-1]
        )
        self.data.drop("TeamName", axis=1, inplace=True)
        # keep only features useful for prediction, work on copy of data
        self.data.drop(
            [
                "index",
                "CaseNumber",
                "Topics",
                "RequesterEmail",
                "EmailObject",
                "LastEmailCCAddress",
                "AttributesURL",
                "ContactAttributesURL",
                "ContactEmail",
                "LastIncomingEmailContent",
                "LastEmailContent",
            ],
            axis=1,
            inplace=True,
        )

        self.data = self.encode_and_bind(self.data, "CMA_in_cc")
        # self.data = self.encode_and_bind(self.data, "TeamName_division")
        # self.data = self.encode_and_bind(self.data, "TeamName_client")
        self.data.drop(columns=["TeamName_division", "TeamName_client"], inplace=True)
        self.data = pd.concat([self.data, self.ext_preds], axis=1)

    def encode_and_bind(self, original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        res = pd.concat([original_dataframe, dummies], axis=1).drop(
            feature_to_encode, axis=1
        )
        return res

    def TrainValidTestSplit(self, train_valid):

        train, valid = train_test_split(
            train_valid, test_size=0.3, random_state=1, stratify=train_valid["Type"]
        )

        X_train = train.loc[:, train.columns != "Type"]
        y_train = train["Type"]

        X_valid = valid.loc[:, valid.columns != "Type"]
        y_valid = valid["Type"]

        return train, valid, X_train, X_valid, y_train, y_valid

    def PlotConfusionMatrix(self, cm):

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt="g", ax=ax)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        plt.show()

        # evaluate model

    def EvaluateModel(self, model, y_pred, y_true):

        self.PlotConfusionMatrix(confusion_matrix(y_true, y_pred))

        return (
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            confusion_matrix(y_true, y_pred),
        )

    def Threshold(self, y_true, y_pred):

        pre, rec, t = precision_recall_curve(y_true, y_pred)

        # get threshold that allows for >90%, >95%, >99% precision
        t_90 = t[next(x for x, val in enumerate(pre) if val >= 0.90)]
        t_95 = t[next(x for x, val in enumerate(pre) if val >= 0.95)]
        t_99 = t[next(x for x, val in enumerate(pre) if val >= 0.99)]

        return [t_90, t_95, t_99]

    def GetResults(self, model, y, pred):
        # create vectors to save results
        precision = np.array([])
        recall = np.array([])
        # understand threshold to have x% precision
        # translate probabilities into predictions
        for t in self.Threshold(y, pred):
            y_pred = pred > t

            # return in-sample precision and recall
            precision_, recall_, conf_mtx_ = self.EvaluateModel(model, y_pred, y)

            # append results
            precision = np.append(precision, precision_)
            recall = np.append(recall, recall_)

        # print results
        results = pd.DataFrame({"Precision": precision, "Recall": recall})
        print(results)

    # logistic regression model
    def LogReg(self, df):

        train, valid, X_train, X_valid, y_train, y_valid = self.TrainValidTestSplit(df)

        # create model
        model = linear_model.LogisticRegression()

        # fit model
        model.fit(X_train, y_train)

        # make predictions
        pred_valid = model.predict_proba(X_valid)[:, 1]

        # get results
        self.GetResults(model, y_valid, pred_valid)

        return model

    def CART(self, df):

        # cross-validation, so we keep the entire train-valid data without further splitting
        # still, split X, y
        X = df.loc[:, df.columns != "Type"]
        y = df["Type"]

        # define parameters of the tree
        tree_para = {
            "criterion": ["gini"],
            "max_depth": range(2, 20, 3),
            "min_samples_split": [2, 5],
            "min_samples_leaf": [2, 5],
        }

        model = GridSearchCV(
            DecisionTreeClassifier(random_state=1), tree_para, cv=5
        )  # verbose = 2 to get messages

        # fit the model
        model.fit(X, y)

        # make predictions
        pred = model.predict_proba(X)[:, 1]

        # get results
        self.GetResults(model, y, pred)

        return model

        # XGBoost

    def XGB(self, df):

        # split X, y
        X = df.loc[:, df.columns != "Type"]
        y = df["Type"]

        # define parameters to try
        """
        xgb_para = {'learning_rate': [0.1],
                    'max_depth':range(2, 8, 2),
                    'min_child_weight':range(1,6,3),
                    'gamma': [i/10.0 for i in range(0,5,2)],
                    }
        """
        xgb_para = {}  # using default parameters

        # define the model
        model = GridSearchCV(
            XGBClassifier(random_state=1), xgb_para, cv=5
        )  # verbose = 2

        # fit the model
        model.fit(X, y)

        # make predictions
        pred = model.predict_proba(X)[:, 1]

        # get results
        self.GetResults(model, y, pred)

        return model

    # Random Forest
    def RF(self, df):

        # split X, y
        X = df.loc[:, df.columns != "Type"]
        y = df["Type"]

        # define parameters to try
        """
        rf_para = {'max_depth':range(3,10,2),
                    'n_estimators': [100, 200, 500, 1000],
                    'min_samples_split': [2, 5],
                    }
        """
        rf_para = {}  # using default parameters

        # define the model
        model = GridSearchCV(
            RandomForestClassifier(random_state=1), rf_para, cv=5, verbose=3
        )  # verbose = 2

        # fit the model
        model.fit(X, y)

        # make predictions
        pred = model.predict_proba(X)[:, 1]

        # get results
        self.GetResults(model, y, pred)

        return model
