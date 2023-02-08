from pyspark.sql import SparkSession
from pyspark.sql.functions import first
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import mean, countDistinct,desc

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
        self.X = self.df.select("text")
        self.y = self.df.select(self.categorical_column)
        print("feature segregation completed")

    def feature_extraction(self):
        print("feature extraction started")
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\s+")
        self.X = tokenizer.transform(self.X)

        hashing_tf = HashingTF(inputCol="tokens", outputCol="tf_features")
        self.X = hashing_tf.transform(self.X)

        idf = IDF(inputCol="tf_features", outputCol="features")
        self.X = idf.fit(self.X).transform(self.X)
        print("feature extraction completed")



df_pyspark=spark.read.option('header','true').csv('data.csv',inferSchema=True)
text_columns = ["Ticket_Title", "Application"] # No text columns in this example
categorical_column = ["Master_SOP"]
model = TextClassificationModel(df_pyspark, text_columns, categorical_column)

model.show_summary()
model.clean_dataset()
model.feature_segregation()
model.feature_extraction()
