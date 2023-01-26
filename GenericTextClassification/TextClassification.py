import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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
        numeric_cols = dataframe.select_dtypes(include=['float64','int64']).columns

        # Handling null values for categorical columns
        for col in categorical_cols:
            dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)

        # Handling null values for numeric columns
        for col in numeric_cols:
            dataframe[col].fillna(dataframe[col].mean(), inplace=True)


    def feature_segregation(self):
        # The error message "Found input variables with inconsistent numbers of samples" is occurring because the number of rows in the multiple text columns you passed in as text_columns is different from the number of rows in the categorical column you passed in as categorical_column.
        # One solution for this is to concatenate the multiple text columns into a single column by using the apply function from pandas and passing in a lambda function that concatenates the values of the two text columns together with a separator in between.
        self.df["text"] = self.df[self.text_columns].apply(lambda x: " ".join(x), axis=1)
        
        
        self.X = self.df["text"]
        self.y = self.df[self.categorical_column]
        #save original lables
        self.label_mapping = dict(enumerate(self.y.unique()))
        
    def feature_extraction(self):
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)

    def train_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(),
            'Linear SVC': LinearSVC()
        }

        self.best_model = None
        self.best_accuracy = 0
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            print(f'{name} accuracy: {accuracy}')
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.best_model = pickle.load(file)

    # def predict(self, text):
    #     text = self.vectorizer.transform([text])
    #     predictions = self.best_model.predict(text)

    #     # if self.categorical_column not in self.df.columns:
    #     #     print("Error: column name not found in the dataframe.")
    #     # else:
    #     #     print('success')
    #     # drop rows with missing values
    #     text = text.copy()
    #     # text.dropna(subset=[self.categorical_column], inplace=True)
    #     # dataframe.dropna(subset=[self.categorical_column], inplace=True)
            
    #     original_labels = [self.label_mapping[label] for label in predictions]
    #     return original_labels

    def predict(self, text):
        text = self.vectorizer.transform([text])
        return self.best_model.predict(text)



# df = pd.read_csv("data.csv")
# model = TextClassificationModel(df, ["Ticket_Title", "Application"], "Master_SOP")
# model.show_summary()
# model.clean_dataset()
# model.feature_segregation()
# model.feature_extraction()
# model.train_model()
# model.save

dataframe = pd.read_csv("data.csv")
text_classification_model = TextClassificationModel(dataframe, ["Ticket_Title", "Application"], "Master_SOP")
# text_classification_model.show_summary()
text_classification_model.clean_dataset()
text_classification_model.feature_segregation()
text_classification_model.feature_extraction()
text_classification_model.train_model()
text_classification_model.save_model("best_model.pkl")
print('model saved')


# print(dataframe.columns)
# Apply the predict function to the DataFrame
# dataframe["predictions"] = text_classification_model.predict(dataframe["text"].to_string())

# Print the results
# print(dataframe)

