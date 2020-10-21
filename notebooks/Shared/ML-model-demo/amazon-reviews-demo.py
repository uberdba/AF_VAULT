# Databricks notebook source
from pyspark.sql.functions import udf, col, count, explode
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType

# COMMAND ----------

# Create dataframe of Amazon Instant Video data
df = spark.sql('''
  SELECT reviewText, overall
  FROM amazon_instant_video
  WHERE reviewText != ""
  LIMIT 5000
''')

display(df)

# COMMAND ----------

# Tokenize reviews
tokenizer = Tokenizer(inputCol="reviewText", outputCol="reviewWords")
words_df = tokenizer.transform(df)
words_df.show()

# COMMAND ----------

# Split into train & test data sets
seed = 0
train_df, test_df = words_df.randomSplit([0.8, 0.2], seed)

# COMMAND ----------

# Find & plot top N words, by count, in train_df
N = 100
train_words_by_count_df = (train_df.withColumn("reviewWords", explode(col("reviewWords")))
                           .groupBy("reviewWords")
                           .agg(count("*").alias("count"))
                           .sort(col("count").desc())
                           .limit(N))
display(train_words_by_count_df)

# COMMAND ----------

# Featurize data by token count
count_vectorizer = CountVectorizer(inputCol="reviewWords", outputCol="rawFeatures", 
                                   vocabSize=N)
count_vectorizer_model = count_vectorizer.fit(train_df)
featurized_train_df = count_vectorizer_model.transform(train_df)
featurized_train_df.show()

# COMMAND ----------

# Review the words that were in the vocabulary
count_vectorizer_model.vocabulary

# COMMAND ----------

# Compute TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_train_df)
rescaled_train_df = (idf_model.transform(featurized_train_df)
                              .withColumnRenamed("overall","label")
                              .select("label", "features"))
rescaled_train_df.show() 

# COMMAND ----------

# Train multinomial logistic regression
lr = LogisticRegression(maxIter=10, family="multinomial")
lr_fit = lr.fit(rescaled_train_df)

# COMMAND ----------

# Predict on test data using multinomial logistic regression
featurized_test_df = count_vectorizer_model.transform(test_df)
rescaled_test_df = (idf_model.transform(featurized_test_df)
                                .withColumnRenamed("overall","label")
                                .select("label", "features"))
predict_test_df = lr_fit.transform(rescaled_test_df)
predict_test_df = predict_test_df.withColumn("label", predict_test_df["label"].cast(DoubleType()))
predict_test_df.select("label", "prediction").show(10)

# COMMAND ----------

# Evaluate predictions
evaluator = MulticlassClassificationEvaluator()
evaluator.setPredictionCol("prediction")
evaluator.evaluate(predict_test_df, {evaluator.metricName: "accuracy"})

# COMMAND ----------

# Show error as histogram
pred