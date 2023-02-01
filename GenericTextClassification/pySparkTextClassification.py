from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, first, mean
from pyspark.ml.classification import LogisticRegression, NaiveBayes, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, NaiveBayes, DecisionTreeClassifier, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, DoubleType


class TextClassificationModel:
    def init(self, dataframe, text_columns, categorical_column):
        self.df = dataframe
        self.text_columns = text_columns
        self.categorical_column = categorical_column

    def show_summary(self):
        print(self.df.count(), len(self.df.columns))
        self.df.printSchema()
        self.df.describe().show()
        self.df.show(5)

    # def clean_dataset(self):
    #     # Handle missing values for numerical columns
    #     self.df = self.df.na.fill(self.df.agg(*[F.avg(c).alias(c) for c in self.df.columns if self.df.schema[c].dataType in [FloatType, DoubleType]]
    #     ).first().asDict())

    #     # Handle missing values for categorical columns
    #     self.df = self.df.na.fill(self.df.agg(
    #         *[F.mode(c).alias(c) for c in self.df.columns if self.df.schema[c].dataType in [StringType]]
    #     ).first().asDict())
     
    def clean_dataset(self):
        # Extract categorical and numeric columns
        categorical_cols = self.df.select(*(F.col(c).cast("string").alias(c) for c in self.df.columns if self.df.dtypes[c] == "string")).columns
        numeric_cols = self.df.select(*(F.col(c).cast("double").alias(c) for c in self.df.columns if self.df.dtypes[c] in ("double", "int"))).columns
        
        # Handling null values for categorical columns
        for col in categorical_cols:
            self.df = self.df.fillna(self.df.groupBy().agg(F.first(col)).collect()[0][0], subset=[col])

        # Handling null values for numeric columns
        for col in numeric_cols:
            self.df = self.df.fillna(self.df.agg({col: 'mean'}).collect()[0][0], subset=[col])


    def feature_segregation(self):
        if len(self.text_columns) > 1:
            self.df = self.df.withColumn("text", F.concat(*[F.col(col) for col in self.text_columns]))
        else:
            self.df = self.df.withColumn("text", F.col(self.text_columns[0]))
        self.X = self.df.select("text")
        self.y = self.df.select(self.categorical_column)

    def feature_extraction(self):
        hashing_tf = HashingTF(inputCol="text", outputCol="rawFeatures", numFeatures=20)
        self.X = hashing_tf.transform(self.X)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        self.X = idf.fit(self.X).transform(self.X)

    

    def train_model(self):
        self.classes_ = self.y.distinct().collect()

        # Create a vector assembler to combine all feature columns into one vector column
        assembler = VectorAssembler(inputCols=["text"], outputCol="features")
        data = assembler.transform(self.df)

        # Split the data into training and testing datasets
        (trainingData, testData) = data.randomSplit([0.8, 0.2])

        self.models = {
            'Logistic Regression': LogisticRegression(featuresCol="features", labelCol="categorical_column"),
            'Naive Bayes': NaiveBayes(featuresCol="features", labelCol="categorical_column"),
            'Decision Tree': DecisionTreeClassifier(featuresCol="features", labelCol="categorical_column"),
            'Random Forest': RandomForestClassifier(featuresCol="features", labelCol="categorical_column"),
            'Linear SVC': LinearSVC(featuresCol="features", labelCol="categorical_column")
        }

        self.best_model = None
        self.best_accuracy = 0
        for name, model in self.models.items():
            model_fit = model.fit(trainingData)
            predictions = model_fit.transform(testData)

            # Evaluate the model using binary classification evaluator for binary classification and multiclass classification evaluator for multiclass classification
            if len(self.classes_) == 2:
                evaluator = BinaryClassificationEvaluator(labelCol="categorical_column")
                accuracy = evaluator.evaluate(predictions)
            else:
                evaluator = MulticlassClassificationEvaluator(labelCol="categorical_column")
                accuracy = evaluator.evaluate(predictions)
            
            print(f'{name} accuracy: {accuracy}')
            print("-------------------------------------")
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model_fit
        print(f'Best Model: {self.best_model}')


    def predict(self, data):
        data = self.vectorizer.transform(data)
        predictions = self.best_model.transform(data)
        prediction_udf = udf(lambda x: float(x.prediction), DoubleType())
        predictions = predictions.withColumn("prediction", prediction_udf(predictions.prediction))
        prediction_schema = StructType([
            StructField("prediction", DoubleType(), True)
        ])
        predictions = predictions.select("prediction").rdd.map(lambda row: row[0]).collect()
        return predictions


from pyspark.sql import SparkSession

# Start a Spark session
spark = SparkSession.builder.appName("DataFrameExample").getOrCreate()

# Read a CSV file into PySpark
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# Show the Spark DataFrame
df.show()






