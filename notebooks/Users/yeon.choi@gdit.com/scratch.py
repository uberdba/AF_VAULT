# Databricks notebook source
# MAGIC %sql
# MAGIC create database yc;

# COMMAND ----------

# MAGIC %sql
# MAGIC convert to delta yc.amazon_instant_video 
# MAGIC optimize yc.amazon_instant_video zorder by (asin)

# COMMAND ----------

# MAGIC %sql
# MAGIC alter table yc.amazon_instant_video partitioned by reviewerID

# COMMAND ----------

# MAGIC %sql
# MAGIC desc extended yc.amazon_instant_video

# COMMAND ----------

dbutils.fs.head('dbfs:/FileStore/tables/Amazon_Instant_Video_5-1.csv')

# COMMAND ----------

dbutils.fs.ls('dbfs:/user/hive/warehouse/yc.db/amazon_instant_video/_delta_log')