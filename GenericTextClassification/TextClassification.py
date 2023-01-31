import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pickle

class TextClassificationModel:
    def __init__(self, dataframe, text_columns, categorical_column):
        self.df = dataframe
        self.text_columns = text_columns
        self.categorical_column = categorical_column

    def show_summary(self):
        print(self.df.info())
        print(self.df.describe())
        print(self.df.head())

    def clean_dataset(self):
        # perform any necessary cleaning of the dataset here

        # Extract categorical and numeric columns
        categorical_cols = dataframe.select_dtypes(include=['object']).columns
        numeric_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns

        # Handling null values for categorical columns
        for col in categorical_cols:
            dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)

        # Handling null values for numeric columns
        for col in numeric_cols:
            dataframe[col].fillna(dataframe[col].mean(), inplace=True)

    def feature_segregation(self):
        if len(self.text_columns) > 1:
            self.df["text"] = self.df[self.text_columns].apply(
                lambda x: " ".join(x), axis=1)
        else:
            self.df["text"] = self.df[self.text_columns[0]]
        self.X = self.df["text"]
        self.y = self.df[self.categorical_column]

    def feature_extraction(self):
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)

    def train_model(self):
        self.classes_ = np.unique(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2)

        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Naive Bayes': MultinomialNB(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Linear SVC': LinearSVC(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier()
        }

        self.best_model = None
        self.best_accuracy = 0
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions, average='weighted',zero_division=1)
            recall = recall_score(self.y_test, predictions, average='weighted',zero_division=1)
            f1 = f1_score(self.y_test, predictions, average='weighted')
            print(f'{name}\n')
            print(f'{name} accuracy: {accuracy}')
            print(f'{name} precision: {precision}')
            print(f'{name} recall: {recall}')
            print(f'{name} f1 score: {f1}')
            print("-------------------------------------")
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
            print(self.best_model)


    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.best_model = pickle.load(file)
            print("----------------",self.best_model)

    def predict(self, data):
        data = self.vectorizer.transform(data)
        predictions = self.best_model.predict(data)
        return predictions

    def predict_original(self, text_data):
        text_data = self.vectorizer.transform(text_data)
        predictions = self.best_model.predict(text_data)
        # print("################",self.X.shape)
        # print("################",text_data.shape)

        if self.X.shape != text_data.shape:
            raise ValueError("Input data shape does not match the shape of the data used to train the model.")
        original_predictions = self.vectorizer.inverse_transform(predictions)
        return original_predictions



dataframe = pd.read_csv("data.csv")
text_classification_model = TextClassificationModel(
    dataframe, ["Ticket_Title", "Application"], "Master_SOP")
text_classification_model.show_summary()
text_classification_model.clean_dataset()
text_classification_model.feature_segregation()
text_classification_model.feature_extraction()
text_classification_model.train_model()
text_classification_model.save_model('best_model.pickle')
text_classification_model.load_model('best_model.pickle')

text_data = dataframe[["Ticket_Title", "Application"]].apply(lambda x: " ".join(x), axis=1)
predictions = text_classification_model.predict(text_data)
print(predictions)

# Add the predictions as a new column in the dataframe
dataframe['predicted_class'] = predictions
print(dataframe['predicted_class'])

# text_data = dataframe[["Ticket_Title", "Application"]].apply(lambda x: " ".join(x), axis=1)
# text_data = text_data.tolist()
# print(type(text_data))
# original_predictions = text_classification_model.predict_original(text_data)
# print(original_predictions)

# text_classification_model.predict(dataframe)
# text_classification_model.save_model("best_model.pkl")
print('completed')

#PREDICT ON a single text data row with two column

import pandas as pd

# create a dataframe with two columns 'Ticket_Title' and 'Application'
data = {'Ticket_Title': ['JDE Incident Request', 'Failed payment batches: JDE to Trax- V2 to V3 folder'],
        'Application': ['GRP-25 JDE EMEA', 'GRP-25 JDE EMEA']}
df = pd.DataFrame(data)

# concatenate the values of the two columns
df['text_data'] = df['Ticket_Title'] + ' ' + df['Application']
print(df)

# predict on the 'text_data' column
predictions = text_classification_model.predict(df['text_data'])

# add the predictions as a new column in the dataframe
df['predictions'] = predictions
print(df['predictions'])

