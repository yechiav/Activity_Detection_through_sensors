# Databricks notebook source
import pandas as pd
import numpy as np
import random
from os import listdir
from os.path import isfile, join
from operator import add
import math
import os
import scipy.stats
from collections import defaultdict
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql import Window
from pyspark.sql.functions import last, first
from pyspark.sql.functions import when
import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
from itertools import chain
from pyspark.sql.functions import create_map, lit
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder,TrainValidationSplit

hyper_Param_tunning=False

seed=42
drop_columns='orientation'

print(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"))

features_list_not_to_Classification=["timestamp","activityID","label"]

summary_Table_Per_Class=pd.DataFrame(columns=["Subject","Class","Messure","Value"])
summary_Table_Per_Subject=pd.DataFrame(columns=["Subject","Messure","Value"])
summary_Table_Cleaning=pd.DataFrame(columns=["Subject","Messure","Value"])

dataSets={}

# dataSets["subject109"]="/FileStore/tables/subject109.dat"
dataSets["subject108"]="/FileStore/tables/subject108.dat"
# dataSets["subject107"]="/FileStore/tables/subject107.dat"
# dataSets["subject106"]="/FileStore/tables/subject106.dat"
# dataSets["subject105"]="/FileStore/tables/subject105.dat"
# dataSets["subject104"]="/FileStore/tables/subject104.dat"
# dataSets["subject103"]="/FileStore/tables/subject103.dat"
# dataSets["subject102"]="/FileStore/tables/subject102.dat"
# dataSets["subject101"]="/FileStore/tables/subject101.dat"
  

resting_HR={}
resting_HR["subject101"]=75
resting_HR["subject102"]=74
resting_HR["subject103"]=68
resting_HR["subject104"]=58
resting_HR["subject105"]=70
resting_HR["subject106"]=60
resting_HR["subject107"]=60
resting_HR["subject108"]=66
resting_HR["subject109"]=54

Label_dict={}
Label_dict[0]="Zero_Remove"
Label_dict[1]="lying"
Label_dict[2]="sitting"
Label_dict[3]="standing"
Label_dict[4]="walking"
Label_dict[5]="running"
Label_dict[6]="cycling"
Label_dict[7]="Nordic_walking"
Label_dict[8]="unknown"
Label_dict[9]="watching_TV"
Label_dict[10]="computer_work"
Label_dict[11]="car_driving"
Label_dict[12]="ascending_stairs"
Label_dict[13]="descending_stairs"
Label_dict[14]="unknown"
Label_dict[15]="unknown"
Label_dict[16]=" vacuum_cleaning"
Label_dict[17]="ironing"
Label_dict[18]="folding_laundry"
Label_dict[19]="house_cleaning"
Label_dict[20]="playing_soccer"
Label_dict[21]="unknown"
Label_dict[22]="unknown"
Label_dict[23]="unknown"
Label_dict[24]="rope_jumping"

schema = StructType([
    StructField("timestamp", DoubleType(), True),
    StructField("activityID", IntegerType(), False),
    StructField("heart_rate", DoubleType(), True),
    StructField("IMU_hand_temperature_1", DoubleType(), True),
    StructField("IMU_hand_3Dacc_2", DoubleType(), True),
    StructField("IMU_hand_3Dacc_3", DoubleType(), True),
    StructField("IMU_hand_3Dacc_4", DoubleType(), True),
    StructField("IMU_hand_3Dacc_5", DoubleType(), True),
    StructField("IMU_hand_3Dacc_6", DoubleType(), True),
    StructField("IMU_hand_3Dacc_7", DoubleType(), True),
    StructField("IMU_hand_3Dgyros_8", DoubleType(), True),
    StructField("IMU_hand_3Dgyros_9", DoubleType(), True),
    StructField("IMU_hand_3Dgyros_10", DoubleType(), True),
    StructField("IMU_hand_3Dmagnet_11", DoubleType(), True),
    StructField("IMU_hand_3Dmagnet_12", DoubleType(), True),
    StructField("IMU_hand_3Dmagnet_13", DoubleType(), True),
    StructField("IMU_hand_orientation_14", DoubleType(), True),
    StructField("IMU_hand_orientation_15", DoubleType(), True),
    StructField("IMU_hand_orientation_16", DoubleType(), True),
    StructField("IMU_hand_orientation_17", DoubleType(), True),
    StructField("IMU_chest_temperature_1", DoubleType(), True),
    StructField("IMU_chest_3Dacc_2", DoubleType(), True),
    StructField("IMU_chest_3Dacc_3", DoubleType(), True),
    StructField("IMU_chest_3Dacc_4", DoubleType(), True),
    StructField("IMU_chest_3Dacc_5", DoubleType(), True),
    StructField("IMU_chest_3Dacc_6", DoubleType(), True),
    StructField("IMU_chest_3Dacc_7", DoubleType(), True),
    StructField("IMU_chest_3Dgyros_8", DoubleType(), True),
    StructField("IMU_chest_3Dgyros_9", DoubleType(), True),
    StructField("IMU_chest_3Dgyros_10", DoubleType(), True),
    StructField("IMU_chest_3Dmagnet_11", DoubleType(), True),
    StructField("IMU_chest_3Dmagnet_12", DoubleType(), True),
    StructField("IMU_chest_3Dmagnet_13", DoubleType(), True),
    StructField("IMU_chest_orientation_14", DoubleType(), True),
    StructField("IMU_chest_orientation_15", DoubleType(), True),
    StructField("IMU_chest_orientation_16", DoubleType(), True),
    StructField("IMU_chest_orientation_17", DoubleType(), True),
    StructField("IMU_ankle_temperature_1", DoubleType(), True),
    StructField("IMU_ankle_3Dacc_2", DoubleType(), True),
    StructField("IMU_ankle_3Dacc_3", DoubleType(), True),
    StructField("IMU_ankle_3Dacc_4", DoubleType(), True),
    StructField("IMU_ankle_3Dacc_5", DoubleType(), True),
    StructField("IMU_ankle_3Dacc_6", DoubleType(), True),
    StructField("IMU_ankle_3Dacc_7", DoubleType(), True),
    StructField("IMU_ankle_3Dgyros_8", DoubleType(), True),
    StructField("IMU_ankle_3Dgyros_9", DoubleType(), True),
    StructField("IMU_ankle_3Dgyros_10", DoubleType(), True),
    StructField("IMU_ankle_3Dmagnet_11", DoubleType(), True),
    StructField("IMU_ankle_3Dmagnet_12", DoubleType(), True),
    StructField("IMU_ankle_3Dmagnet_13", DoubleType(), True),
    StructField("IMU_ankle_orientation_14", DoubleType(), True),
    StructField("IMU_ankle_orientation_15", DoubleType(), True),
    StructField("IMU_ankle_orientation_16", DoubleType(), True),
    StructField("IMU_ankle_orientation_17", DoubleType(), True)])

# COMMAND ----------



def Print_head(df,additional=False):
  if(additional==False):
    temp = df.select("timestamp", "activityID", "heart_rate")
  else:
    temp = df.select("timestamp", "activityID", "heart_rate",additional)
  print(temp.show(20))

def load_data(Title):
  df = spark.read.format("csv").option("header", "false").option("delimiter", " ").schema(schema).load(dataSets[Title])
  return df

def Clean_Data(df,name):
  global summary_Table_Cleaning
  df_columns = summary_Table_Cleaning.columns
  summary_Table_Cleaning = summary_Table_Cleaning.append(pd.DataFrame([[name,"All_Values",df.count()]],columns=df_columns),ignore_index=True)

#   print(df.count())
  print("count of all " + str(df.count()))
  print("count of activityID = 0 " + str(df.filter("activityID = 0").count()))
  df = df.filter("activityID != 0")
  summary_Table_Cleaning = summary_Table_Cleaning.append(pd.DataFrame([[name,"After_activityID=0_Remove",df.count()]],columns=df_columns),ignore_index=True)

  #remove "Orientation" columns
  all_columns = df.columns
  Cols_remove = [s for s in all_columns if drop_columns in s]
  df = df.drop(*Cols_remove)

  return df

def fill_activity_na(df):
  # define the window
  window = Window.orderBy('timestamp').rowsBetween(-20, 0)

  # define the forward-filled column
  filled_column = last(df['heart_rate'], ignorenulls=True).over(window)

  df = df.withColumn('heart_rate', filled_column)

  return df

def PreProcess(df,name):
  global summary_Table_Cleaning
  #replace NaN with Null
  cols = [F.when(~F.col(x).isin("NULL", "NA", "NaN"), F.col(x)).alias(x)  for x in df.columns]
  df = df.select(*cols)

  #fill hearbeat values
  df = fill_activity_na(df)
  
  
  df = df.withColumn("heart_rate_menus_rest", df.heart_rate-resting_HR[name])
  Print_head(df,"heart_rate_menus_rest")
  
  #replace label with names
  mapping = create_map([lit(x) for x in chain(*Label_dict.items())])

  df = df.withColumn("label", mapping[df["activityID"]])
  Print_head(df,"label")
  df.groupBy('label').count().show()
  
  
  #Drop all null
  print("before Null remove - count of all " + str(df.count()))
  df = df.dropna()
  print("After Null remove - count of all " + str(df.count()))
  
  df_columns = summary_Table_Cleaning.columns
  summary_Table_Cleaning = summary_Table_Cleaning.append(pd.DataFrame([[name,"AfterNaNRemove",df.count()]],columns=df_columns),ignore_index=True)
  
  #addnig avg tmp
  df = df.withColumn("Avg_temperature",(df.IMU_hand_temperature_1+df.IMU_chest_temperature_1+df.IMU_ankle_temperature_1)/3)

  return df


def Train_Model(df,name):

  print("------------------------{}---------------------------------".format(name))
  features_list = list(set(df.columns)-set(features_list_not_to_Classification))
  
  labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)

  vec_assembler = VectorAssembler(inputCols=features_list, outputCol="indexedFeatures")

  # Split the data into training and test sets (30% held out for testing)
  (trainingData, testData) = df.randomSplit([0.8, 0.2])

  # Train a RandomForest model.
  rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=100, minInfoGain=0.01,maxDepth=5)

  # Convert indexed labels back to original labels.
  labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                 labels=labelIndexer.labels)

  # Chain indexers and forest in a Pipeline
  pipeline = Pipeline(stages=[labelIndexer, vec_assembler, rf, labelConverter])

  # Train model.  This also runs the indexers.
  model = pipeline.fit(trainingData)
  # Make predictions.
  predictions = model.transform(testData)

  return model,predictions
  

def Evaluate_model(model,df,predictions,name):
  global summary_Table_Per_Class
  global summary_Table_Per_Subject
  
    #-----------------------------Evalution-------------------------------------------
  features_list = list(set(df.columns)-set(features_list_not_to_Classification))
  present_cols = list(set(predictions.columns)-set(features_list))
  predictions.select(present_cols).show(5)
  

  predictions.groupBy(['predictedLabel','label']).count().show()
  display(predictions.groupBy(['predictedLabel','label']).count())
  #index to label
  indx_2_label_Temp = predictions.groupBy(['indexedLabel','label']).count()
  index_2_label2 = indx_2_label_Temp.toPandas().set_index('indexedLabel').T.to_dict('list')

  # Select (prediction, true label) and compute test error
  evaluator = MulticlassClassificationEvaluator(
      labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
  accuracy = evaluator.evaluate(predictions)
  
  print("Overhuall preformance:")
  print("Test Error = %g" % (1.0 - accuracy))
  print("accuracy = %g" % accuracy)
  rfModel = model.stages[2]
  print(rfModel)  # summary only

  predictionAndLabels = predictions.select(col('prediction'), col('indexedLabel'))
  
  predictionAndLabels.printSchema()
  tp = predictionAndLabels.rdd.map(tuple)
#   print(tp.take(20))
  metrics = MulticlassMetrics(tp)
  
    # Overall statistics
  precision = metrics.precision()
  recall = metrics.recall()
  f1Score = metrics.fMeasure()
  print("Summary Stats")
  print("Precision = %s" % precision)
  print("Recall = %s" % recall)
  print("F1 Score = %s" % f1Score)

  predictionAndLabels.printSchema()
  # Statistics by class
  labels = predictionAndLabels.select(col('indexedLabel')).distinct().collect()
  for label in sorted(labels):
    class_name=index_2_label2[label['indexedLabel']][0]
    df_columns = summary_Table_Per_Class.columns
    summary_Table_Per_Class = summary_Table_Per_Class.append(pd.DataFrame([[name,class_name,"precision",metrics.precision(label['indexedLabel'])]],columns=df_columns),ignore_index=True)
    summary_Table_Per_Class = summary_Table_Per_Class.append(pd.DataFrame([[name,class_name,"recall",metrics.recall(label['indexedLabel'])]],columns=df_columns),ignore_index=True)
    summary_Table_Per_Class = summary_Table_Per_Class.append(pd.DataFrame([[name,class_name,"F1",metrics.fMeasure(label['indexedLabel'], beta=1.0)]],columns=df_columns),ignore_index=True)
    
  
  df_columns = summary_Table_Per_Subject.columns
  summary_Table_Per_Subject = summary_Table_Per_Subject.append(pd.DataFrame([[name,"Accuracy",accuracy]],columns=df_columns),ignore_index=True)
  summary_Table_Per_Subject = summary_Table_Per_Subject.append(pd.DataFrame([[name,"Recall",metrics.weightedRecall]],columns=df_columns),ignore_index=True)
  summary_Table_Per_Subject = summary_Table_Per_Subject.append(pd.DataFrame([[name,"precision",metrics.weightedPrecision]],columns=df_columns),ignore_index=True)
  summary_Table_Per_Subject = summary_Table_Per_Subject.append(pd.DataFrame([[name,"F1_Score",metrics.weightedFMeasure()]],columns=df_columns),ignore_index=True)
  summary_Table_Per_Subject = summary_Table_Per_Subject.append(pd.DataFrame([[name,"F0_5_Score",metrics.weightedFMeasure(beta=0.5)]],columns=df_columns),ignore_index=True)
  summary_Table_Per_Subject = summary_Table_Per_Subject.append(pd.DataFrame([[name,"False_Pos_Rate",metrics.weightedFalsePositiveRate]],columns=df_columns),ignore_index=True)

  
    #----------------------------Feature Importnace------------------------------------

  data_frame_columns = features_list
  feature_importance = rfModel.featureImportances
  
  model_i = pd.DataFrame(feature_importance.toArray(), columns=["values"])  
  features_col = pd.Series(data_frame_columns)  
  model_i["features"] = features_col  
  sort_by_importance = model_i.sort_values('values',ascending=False)
  print("By feature importance")
  print(sort_by_importance)
  
  return 


def Hyper_Param_tunning(df,name):
  features_list = list(set(df.columns)-set(features_list_not_to_Classification))
  
  labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)

  vec_assembler = VectorAssembler(inputCols=features_list, outputCol="indexedFeatures")

  # Train a RandomForest model.
  rf = RandomForestClassifier()

  # Convert indexed labels back to original labels.
  labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                 labels=labelIndexer.labels)

  pipeline = Pipeline(stages=[labelIndexer, vec_assembler, rf, labelConverter])
    
  paramGrid = ParamGridBuilder().addGrid(rf.labelCol, ["indexedLabel"])\
                                .addGrid(rf.featuresCol, ["indexedFeatures"])\
                                .addGrid(rf.maxDepth, [0, 1, 5])\
                                .addGrid(rf.minInfoGain, [0.01, 0.001])\
                                .addGrid(rf.numTrees, [5, 10, 30,100])\
                                .build()
        
  evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
  
  valid = CrossValidator(estimator=pipeline,
                             estimatorParamMaps=paramGrid,
                             evaluator=evaluator,
                             numFolds=5)
  model = valid.fit(df)
  result = model.bestModel.transform(df)
  
  print("BestModel")
  rfModel = model.bestModel.stages[2].extractParamMap()
  print(rfModel)
  
  return model.bestModel,result

def display_resutls(mode="Per_class"):
  global summary_Table_Per_Class
  global summary_Table_Per_Subject
  global summary_Table_Cleaning
  
  if(mode=="Per_class"):
    df1 = spark.createDataFrame(summary_Table_Per_Class)
    display(df1)
  if(mode=="Per_Subject"):
    df2 = spark.createDataFrame(summary_Table_Per_Subject)
    display(df2)
  if(mode=="Cleaning"):
    df3 = spark.createDataFrame(summary_Table_Cleaning)
    display(df3)
  return 


def Main_fun(Title):
  global hyper_Param_tunning

  
  df = load_data(Title)
  df = Clean_Data(df,Title)
  df = PreProcess(df,Title)

  if(hyper_Param_tunning):
    model,predictions = Hyper_Param_tunning(df,Title)
  else:
    model,predictions = Train_Model(df,Title)
    Evaluate_model(model,df,predictions,Title)

  return 0

for Title in dataSets:
  Main_fun(Title)



# COMMAND ----------

display_resutls("Cleaning")

# COMMAND ----------

display_resutls("Per_Subject")

# COMMAND ----------

display_resutls("Per_class")
