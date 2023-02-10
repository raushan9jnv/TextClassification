from pyspark.sql import SparkSession
from pyspark.sql.functions import first
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import mean, countDistinct,desc
from pyspark.sql.functions import lit
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.sql.functions import concat_ws
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, NaiveBayes, DecisionTreeClassifier, RandomForestClassifier, LinearSVC
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import OneHotEncoder, StringIndexer


spark = SparkSession.builder.appName("TextClassificationModel").getOrCreate()

class TextClassificationModel:
    def __init__(self, dataframe, text_columns, categorical_column):
        self.df = dataframe
        self.text_columns = text_columns
        self.categorical_column = categorical_column
        
    def show_summary(self):
        print("DataFrame Schema:")
        self.df.printSchema()
        print("\nDataFrame Summary:")
        self.df.describe().show()
        print("\nDataFrame Head:")
        self.df.show(5)

 

    def clean_dataset(self):
        # perform any necessary cleaning of the dataset here

        # Extract categorical and numeric columns
        categorical_cols = [col for col in self.df.dtypes if col[1] == "string"]
        numeric_cols = [col for col in self.df.dtypes if col[1] in ["double", "int"]]

        # Handling null values for categorical columns
        for col, _ in categorical_cols:
            mode_value = self.df.groupBy().agg(countDistinct(col).alias("count")).sort(desc("count")).limit(1).collect()[0][0]
            self.df = self.df.fillna(mode_value, subset=[col])

        # Handling null values for numeric columns
        for col, _ in numeric_cols:
            mean_value = self.df.agg(mean(col)).first()[0]
            self.df = self.df.fillna(mean_value, subset=[col])

        return self.df

    def feature_segregation(self):
        print("feature segregation started")
        if len(self.text_columns) > 1:
            self.df = self.df.withColumn("text", concat_ws(" ", *self.text_columns))
        else:
            self.df = self.df.withColumnRenamed(self.text_columns[0], "text")
        # self.X = self.df.select("text")
        self.df = self.df.select("text")
        # self.y = self.df.select(self.categorical_column)

        # # Convert the non-numeric column to a numeric format
        # string_indexer = StringIndexer(inputCol='Master_SOP', outputCol='Master_SOP_Index')
        # model = string_indexer.fit(self.y)
        # self.y = model.transform(self.y)
        

        # # Drop the original string column and keep the numerical column
        # self.y = self.y.drop("Master_SOP").withColumnRenamed("Master_SOP_Index", "Master_SOP")

        # # Create a vector assembler to combine all feature columns into one vector column
        # assembler_y = VectorAssembler(inputCols=["Master_SOP"], outputCol="Master_SOP_vector")
        # self.y = assembler_y.transform(self.y)
        # print(self.y.show())
        print("feature segregation completed")

    def feature_extraction(self):
        print("feature extraction started")
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\s+")
        # self.X = tokenizer.transform(self.X)
        self.df = tokenizer.transform(self.df)

        hashing_tf = HashingTF(inputCol="tokens", outputCol="tf_features")
        # self.X = hashing_tf.transform(self.X)
        self.df = hashing_tf.transform(self.df)

        idf = IDF(inputCol="tf_features", outputCol="features")
        # self.X = idf.fit(self.X).transform(self.X)
        self.df = idf.fit(self.df).transform(self.df)
        print(self.df)
        print("feature extraction completed")

    def train_model(self):
        # # self.classes_ = self.y.distinct().collect()

        # # Create a vector assembler to combine all feature columns into one vector column
        # assembler = VectorAssembler(inputCols=["features"], outputCol="vector_features")
        # data = assembler.transform(self.X)


        # # Alternatively, if there is no common column, you can join on the row index
        # data = data.withColumn("index", monotonically_increasing_id())
        # self.y = self.y.withColumn("index", monotonically_increasing_id())
        # data = data.join(self.y, on=["index"])
    
        # # Drop the index column if it was added
        # data = data.drop("index")

        # print("######################################")
        # # print(data.show())
        # # print(data.limit(5).show())
        # # print(self.y.show())

        # self.classes_ = self.y.distinct().collect()

        # Convert the categorical column "self.y" into numeric
        indexer = StringIndexer(inputCol=self.categorical_column, outputCol="categorical_index")
        print(self.df)
        data = indexer.fit(self.df).transform(self.df)
        # print(data.show())

        # Create a vector assembler to combine all feature columns into one vector column
        # assembler = VectorAssembler(inputCols=["text"], outputCol="features")
        assembler = VectorAssembler(inputCols=["features"], outputCol="vector_features")
        data = assembler.transform(data)


        # Split the data into training and testing datasets
        (trainingData, testData) = data.randomSplit([0.8, 0.2])
        print(self.categorical_column)
        print(self.categorical_column)

        self.models = {
            # 'Logistic Regression': LogisticRegression(featuresCol="vector_features"),
            # 'Naive Bayes': NaiveBayes(featuresCol="vector_features"),
            'Decision Tree': DecisionTreeClassifier(featuresCol="vector_features", labelCol="Master_SOP"),
            # 'Random Forest': RandomForestClassifier(featuresCol="vector_features", labelCol=self.categorical_column),
            # 'Linear SVC': LinearSVC(featuresCol="vector_features", labelCol=self.categorical_column)
        }

        self.best_model = None
        self.best_accuracy = 0
        for name, model in self.models.items():
            # Train the model
            model = model.fit(trainingData)

            # Evaluate the model on test data
            predictions = model.transform(testData)
            accuracy = predictions.filter(predictions[self.categorical_column] == predictions["prediction"]).count() / predictions.count()

            # Save the best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
        
        print(f"Best Model: {list(self.models.keys())[list(self.models.values()).index(self.best_model)]} with accuracy: {self.best_accuracy}")



df_pyspark=spark.read.option('header','true').csv('data.csv',inferSchema=True)
text_columns = ["Ticket_Title", "Application"] # No text columns in this example
categorical_column = "Master_SOP"
model = TextClassificationModel(df_pyspark, text_columns, categorical_column)

model.show_summary()
# model.clean_dataset()
model.feature_segregation()
model.feature_extraction()
model.train_model()
