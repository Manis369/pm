# Databricks notebook source
!pip install --upgrade scikit-learn imbalanced-learn
!pip install -U imbalanced-learn

# COMMAND ----------

!pip install smote-variants

# COMMAND ----------

import pm_utils



dbutils.widgets.dropdown("bw_name", "BWP", ["BWP", "BWT", "BWV"], "Select Relevant BW")
dbutils.widgets.text("bw_username", "C5368226", "BW Username")
dbutils.widgets.text("bw_scope_name", "C5368226", "BW Scope Name")
dbutils.widgets.text("bw_scope_key", "C5368226_BWP", "BW Scope Key Name")
dbutils.widgets.dropdown("type", "generic", ["generic", "consulting"], "Model Type")

# COMMAND ----------

# MAGIC %md ##### Stats
# MAGIC
# MAGIC ###### SA vs PF tables
# MAGIC - (PF) CL_CORP_MD_HIER_PORTFOLIO_CY_FLAT_ALL = 26611
# MAGIC - (SA) CL_CORP_MADA_HIER_MATERIAL_FLAT = 27122
# MAGIC - Difference = 511
# MAGIC
# MAGIC ###### Pipeline Data
# MAGIC Last 40 months data:
# MAGIC
# MAGIC - (PF) CL_EXAN_PM_PIPELINE_ALL: 17459286 / 17460372
# MAGIC - (SA) Custom Pipeline Join: 18115470
# MAGIC - Difference = 656,184
# MAGIC - (PF) Custom Pipeline Join (5-Levels): 14549685
# MAGIC - (PF) Custom Pipeline Join (7-Levels): 20370007
# MAGIC - (PF) Custom Pipeline Join (2-7 Levels): 17460156 (Diff with original View: 870)
# MAGIC - (PF) Consulting row count: 1143906
# MAGIC - (SA) Consulting row count: 1153392
# MAGIC - (PF_Consulting) Top X products: 148
# MAGIC - (SA_Consulting) Top X products: 105
# MAGIC - (SA_Generic) Top X products: 649
# MAGIC
# MAGIC ###### Products Sold
# MAGIC - In last 30 months, all unique products sold across all SAs: 1236
# MAGIC - In last 30 months, unique products where sold count is greater than 50: 670

# COMMAND ----------

# MAGIC %md ##### Init spark, adls client and bw credentials

# COMMAND ----------

from azure.datalake.store import core, lib
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity  import ClientSecretCredential

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from matplotlib import pyplot as plt
from numpy import where
import os
import re
import tempfile
import smote_variants as sv
from dataclasses import dataclass
from functools import reduce
from datetime import datetime

import pandas as pd
import numpy as np
from joblib import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV       
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, recall_score

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

from pyspark.sql import DataFrame
from pyspark.sql.functions import year, month, add_months, quarter, lit, concat, datediff, to_date, rand, Column, array, arrays_zip, explode, col, trim, countDistinct, count, expr, when, isnan, udf
from datetime import date
from pyspark.sql.types import StringType
from dateutil.relativedelta import relativedelta

install_cluster_dependencies()

spark = init_spark_with_adls_cred(spark)
client = get_corp_datalake_client()
bw_credentials = get_bw_credentials(
  bw_instance = dbutils.widgets.get("bw_name"),
  username = dbutils.widgets.get("bw_username"),
  password = dbutils.secrets.get(
    scope = dbutils.widgets.get("bw_scope_name"), key = dbutils.widgets.get("bw_scope_key"))
)

model_type = dbutils.widgets.get("type")
print(f"Model type is {model_type}")

#base_adls_storage_path = "/mde/prod/EXAN/data/store/PropensityModels/models/2023"
base_adls_storage_path = "store/PropensityModels/2024/BP/Aug"

adls_storage_path = f"{base_adls_storage_path}/models/{model_type}/"
print(f"Storage path on ADLS for current model type is {adls_storage_path}")

adls_features_storage_path = f"{base_adls_storage_path}/{model_type}/features"
print(f"Storage path on ADLS for selected features is {adls_features_storage_path}")

adls_accuracy_folder_path = f"{base_adls_storage_path}/models/{model_type}/accuracy"
print(f"Storage path for accuracy scores is {adls_accuracy_folder_path}")

SOLUTION_AREA_HIERARCHY_YEAR = "2024"
print(f"Solution area hierarchy used is: {SOLUTION_AREA_HIERARCHY_YEAR}")

# COMMAND ----------

# MAGIC %md ##### Data extraction

# COMMAND ----------

# MAGIC %md ###### Master Data

# COMMAND ----------

masterdata = read_master_data()
masterdata = masterdata.drop('PPLNENTY_FINAL')

# COMMAND ----------

# MAGIC %md ###### Pipeline data with Solution Area

# COMMAND ----------

sa_flat_df = get_solution_area_data(SOLUTION_AREA_HIERARCHY_YEAR)
pipeline_data_with_sa = (get_pipeline_with_sa(sa_flat_df)
                          .select("IC_ACCOUNT_ID", "IC_OBJECT_ID", "IC_STATUS", "IC_STATUS_SINCE", "CA_STATUS_SINCE_DATE",
                                  "LEVEL_CODE", "PBUP_AC", "LEVEL_NAME", "CREATED_AT_DATE", "CREATED_AT")
                          .withColumn("CA_STATUS_SINCE_DATE", col("CA_STATUS_SINCE_DATE").cast("date"))
                          .withColumn("CREATED_AT_DATE", col("CREATED_AT_DATE").cast("date"))
                          .filter(trim(col("LEVEL_CODE")) != '')
                        )
print(f"All pipeline records: {pipeline_data_with_sa.count()}")

pipeline = (pipeline_data_with_sa
            .filter(pipeline_data_with_sa["CREATED_AT"] > get_datetime_in_past(-42))
            .withColumn("IC_STATUS_SINCE", pipeline_data_with_sa.IC_STATUS_SINCE.cast('integer')))
pipeline.cache()

print(f"Pipeline records created in last 42 months: {pipeline.count()}")
#pipeline.show(20, False)

# COMMAND ----------

# MAGIC %md #### Data preparation

# COMMAND ----------

# MAGIC %md ###### Top 50 purchased products in 30 months

# COMMAND ----------

# Top 50 products purchased
# Note original code which calculated -30 months was wrong due to which the entries returned were higher than this function returns
def get_top_50_purchased_products(df: DataFrame) -> DataFrame:
  top_products_purchased = (df
            .filter(col("CREATED_AT") > get_datetime_in_past(-30))
            .filter((col("IC_STATUS") == 'E0003') | (col("IC_STATUS") == 'E0004')) # E0003 = won, E0004 = booked (Purchased oppurunity)
            .sort(col("LEVEL_NAME"), col("LEVEL_CODE"))
            .groupBy(col("LEVEL_CODE"), col("LEVEL_NAME"))
            .agg(countDistinct(col("IC_ACCOUNT_ID")))
            .alias('std')
            .sort(col("count(IC_ACCOUNT_ID)"), ascending=False))

  print(top_products_purchased.count())
  return top_products_purchased.filter(col("count(IC_ACCOUNT_ID)") > 49) 

if model_type == "consulting":
  top_50_products_purchased = get_top_50_purchased_products(pipeline_cons)
else:
  top_50_products_purchased = get_top_50_purchased_products(pipeline)

print(f"Model type is {model_type} and total top X products are {top_50_products_purchased.count()}")
display(top_50_products_purchased)


# COMMAND ----------

top_50_products_purchased.filter(col("LEVEL_CODE") == '').count()

# COMMAND ----------

# MAGIC %md ###### (Labels) Extract all possible dependent variables

# COMMAND ----------

# All purchased products per LoB (Planning Entity) in last 8 quaters
# The output variable DVs, has only 1 in the DV column
def get_purchased_products_per_pentity(df: DataFrame, quater_number: int) -> DataFrame:
  return (df
    .filter((col("IC_STATUS") == 'E0003') | (col("IC_STATUS") == 'E0004'))
    .filter(col("IC_STATUS_SINCE") >= get_date_in_past(-3 * quater_number)) # Quater Start Date
    .filter(col("IC_STATUS_SINCE") < get_date_in_past(-3 * (quater_number - 1))) # Quater End Date
    .groupBy("PBUP_AC", "LEVEL_NAME", "LEVEL_CODE")
    .count()
#    .withColumn("QuaterStart", lit(get_date_in_past(-3 * quater_number)))
#    .withColumn("QuaterEnd", lit(get_date_in_past(-3* (quater_number-1))))
    .withColumn("Period", lit(quater_number))
    .withColumn("DV", lit(1)))

series = []
for quarter_number_index in range(1, 9):
  if model_type == "consulting":
    print(f"Model type is {model_type}, hence using consulting pipeline data")
    series.append(get_purchased_products_per_pentity(pipeline_cons, quarter_number_index))
  else:
    print(f"Model type is {model_type}, hence using all pipeline data")
    series.append(get_purchased_products_per_pentity(pipeline, quarter_number_index))

joined_DVs = reduce(DataFrame.unionAll, series)

DVs = joined_DVs.drop_duplicates()

# COMMAND ----------

# MAGIC %md ##### (Features) Independent variables

# COMMAND ----------

# MAGIC %md ###### Count of open opportunities for LEVEL5 in last 8 quaters

# COMMAND ----------

# Shift by one quater here. To find which products were converted from Open to Booked. Also to find the profile of account in previous quater.

# All products per LoB (Planning Entity) which are Open in last 8 quaters (skipping first quater)
# Returns new opportunity per LEVEL_CODE in last 8 quaters
def get_new_opportunity_per_pentity_per_product(quater_number: int):
  return (pipeline
    .filter(pipeline["LEVEL_NAME"] == 'LEVEL6') # PF Level 4 = SA Level 5, PF Logical Product (L5) = SA Logical Product (L6)
    .filter((pipeline["IC_STATUS"] == 'E0001') | (pipeline["IC_STATUS"] == 'E0007')) # E0001 = New, E0007 = Inprogress (Open opputunity)
    .withColumn("LEVEL_CODE", concat(lit('OpenOpps_3months_L6_'), pipeline.LEVEL_CODE))
    .filter(pipeline["IC_STATUS_SINCE"] >= get_date_in_past(-3 * (quater_number + 1))) # Quater Start Date
    .filter(pipeline["IC_STATUS_SINCE"] < get_date_in_past(-3 * (quater_number))) # Quater End Date
    .groupBy("PBUP_AC", "LEVEL_NAME", "LEVEL_CODE")
    .count()
    .withColumn("Period", lit(quater_number))
    .withColumn("counter", lit(1)))

series_l6_open_opps = []
for quarter_number_index in range(1, 9):
  series_l6_open_opps.append(get_new_opportunity_per_pentity_per_product(quarter_number_index))

OpenQuarter_trans = reduce(DataFrame.unionAll, series_l6_open_opps).drop_duplicates()
OpenQuarter = OpenQuarter_trans.groupBy(['PBUP_AC', 'Period']).pivot('LEVEL_CODE').sum('count')


# COMMAND ----------

# MAGIC %md ###### Count of open opportunities for LEVEL5 in last 8 quaters (shifted by 12 months)

# COMMAND ----------

# Takes one year of data while going back quaters
# -15 to -3
# -18 to -6
# -21 to -9

# Find 8 quaters shifted by 1 year (12 months)
# All products per LoB (Planning Entity) which are Open in last 8 quaters (shifted by one year)
def get_new_opportunity_per_pentity_per_product_12_month_shift(quater_number: int):
  return (pipeline
    .filter(pipeline["LEVEL_NAME"] == 'LEVEL6')
    .filter((pipeline["IC_STATUS"] == 'E0001') | (pipeline["IC_STATUS"] == 'E0007'))
    .withColumn("LEVEL_CODE", concat(lit('OpenOpps_12months_L6_'), pipeline.LEVEL_CODE))
    .filter(pipeline["IC_STATUS_SINCE"] >= get_date_in_past(-3 * quater_number - 12, -3 * (quater_number + 1))) # Year Start Date (Go back 1 year)
    .filter(pipeline["IC_STATUS_SINCE"] < get_date_in_past(-3 * quater_number)) # Year End Date
    .groupBy("PBUP_AC", "LEVEL_CODE")
    .count()
    .withColumn("Period", lit(quater_number))
    .withColumn("counter", lit(1)))


series_l6_yearly_open_opps = []
for quater_number_index in range(1, 9):
  series_l6_yearly_open_opps.append(get_new_opportunity_per_pentity_per_product_12_month_shift(quater_number_index))

OpenYear_trans = reduce(DataFrame.unionAll, series_l6_yearly_open_opps).drop_duplicates()
OpenYear = OpenYear_trans.groupBy(['PBUP_AC', 'Period']).pivot('LEVEL_CODE').sum('count')

# COMMAND ----------

# MAGIC %md ###### Days since first won/booked opportunity (LEVEL6)

# COMMAND ----------

from datetime import datetime
def get_days_since_open_per_pentity_per_level_code(quater_number: int):
  FirstWB_trans = (
    pipeline
      .filter(pipeline["LEVEL_NAME"] == 'LEVEL6')
      .filter((pipeline["IC_STATUS"] == 'E0001') | (pipeline["IC_STATUS"] == 'E0007'))
      .withColumn("LEVEL_CODE", concat(lit('First_WB_Opp_L6_'), pipeline.LEVEL_CODE))
      .filter(pipeline["IC_STATUS_SINCE"] > 19000101)
      .filter(pipeline["IC_STATUS_SINCE"] < get_date_in_past(-3 * (quater_number)))
      .groupBy("PBUP_AC", "LEVEL_CODE")
      .agg({'CA_STATUS_SINCE_DATE': 'min'}) # First time open date
      .withColumnRenamed("min(CA_STATUS_SINCE_DATE)", "CA_STATUS_SINCE_DATE")
      .withColumn("Period", lit(quater_number))
      .withColumn("Quarter_Start_date", lit(get_date_in_past(-3 * (quater_number))))
  )

  FirstWB_trans = (
    FirstWB_trans
      .withColumn("diff",
          datediff(
            to_date(lit(datetime.strptime(str(get_date_in_past(-3 * (quater_number))), "%Y%m%d")), 'yyyyMMdd'),
            FirstWB_trans["CA_STATUS_SINCE_DATE"])))
  return FirstWB_trans
  

series_wb_l6_opps = []
for quater_number_index in range(1, 9):
  series_wb_l6_opps.append(get_days_since_open_per_pentity_per_level_code(quater_number_index))

FirstWB_trans = reduce(DataFrame.unionAll, series_wb_l6_opps).drop_duplicates()
FirstWB = FirstWB_trans.groupBy(['PBUP_AC', 'Period']).pivot('LEVEL_CODE').mean('diff')

# COMMAND ----------

# MAGIC %md ##### Cache all features and labels

# COMMAND ----------

DVs.cache().count()
masterdata.cache().count()
OpenQuarter.cache().count()
OpenYear.cache().count()
FirstWB.cache().count()

# COMMAND ----------

# MAGIC %md ### Model Creation

# COMMAND ----------

# MAGIC %md ###### Select products whose model has to be created

# COMMAND ----------

cust_50 = top_50_products_purchased.toPandas()

if model_type == "consulting":
    print(f"Model type is {model_type} and filtering top X customers on Level 4/5/6/7")
    print(f"Total products before filtering are {len(cust_50)}")
    cust_50 = cust_50[((cust_50.LEVEL_NAME == "LEVEL4") | (cust_50.LEVEL_NAME == "LEVEL5") | (cust_50.LEVEL_NAME == "LEVEL6") | (cust_50.LEVEL_NAME == "LEVEL7"))
                      & (cust_50.LEVEL_CODE.notnull())]
    print(f"Total products after filtering are {len(cust_50)}")
else:
    # In solution area hierarchy, checked the products on Level7 and there are none, so basically, we only consider level 5 and 6
    print(f"Model type is {model_type} and filtering top X customers on Level 6")
    print(f"Total products before filtering are {len(cust_50)}")
    cust_50 = cust_50[((cust_50.LEVEL_NAME == "LEVEL6") | (cust_50.LEVEL_NAME == "LEVEL7"))
                      & (cust_50.LEVEL_CODE.notnull())]
    print(f"Total products after filtering are {len(cust_50)}")

cust_50 = cust_50.sort_values('LEVEL_CODE').reset_index()
cust_50 = cust_50.sort_values('count(IC_ACCOUNT_ID)', ascending = False)

# COMMAND ----------

display(cust_50[["LEVEL_CODE", "LEVEL_NAME"]])

# COMMAND ----------

# MAGIC %md ##### Accuracy Check

# COMMAND ----------

# MAGIC %md ###### Train/test split, Fit, Evaluation Code

# COMMAND ----------

def accuracy_calculate(clf_final, X_test_net, y_test, adls_client: core.AzureDLFileSystem, solution: str, level: str, adls_storage_path):

  predictions = clf_final.predict(X_test_net)

  accuracy_score_output = accuracy_score(y_test, predictions)
  classification_report_output = classification_report(y_test, predictions)

  accuracy_score_output = accuracy_score(y_test, predictions)
  classification_report_output = classification_report(y_test, predictions, output_dict = True)
  auc = roc_auc_score(y_test, predictions)

  accuracy_output = f"Solution: {solution}\nLevel: {level}\nAcc: {accuracy_score_output}\nauc: {auc}\nClassification Report: {classification_report_output}"
  
  def write_to_file(dump_file: str, date: str):
    with open(dump_file, 'wt') as fw:
        fw.write(date)

  _, path = tempfile.mkstemp()
  write_to_file(path, accuracy_output)
  remote_path = os.path.join(adls_accuracy_folder_path, f"{Solution}.acc")
  print(f"Write accuracy report to {remote_path}")

  #client.put(path, remote_path) # Gen1
  
  # Read local file
  with open(path, 'r', encoding="utf-8") as local_file_content:
    file_contents = local_file_content.read()

    file_client = client.get_file_client(remote_path)
    file_client.upload_data(data=file_contents, overwrite=True)

  os.remove(path)

# COMMAND ----------

def correlation(dataset, threshold):  
  corr_matrix = dataset.corr()
  # Remove line using np.tril
  upper = corr_matrix[corr_matrix.columns]  # Select upper triangle directly
  col_corr = set(col for col, row in upper.iteritems() for other_col in row.index if col != other_col and abs(row[other_col]) > threshold)
  return col_corr

# COMMAND ----------

def smotetomek(X_train_net, y_train):
  try:
    print("The number of classes before fit {}".format(Counter(y_train)))
    smt = SMOTETomek(random_state=42)
    X_train_ns, y_train_ns = smt.fit_resample(X_train_net, y_train)
    print("The number of classes after fit {}".format(Counter(y_train_ns)))
    return X_train_ns, y_train_ns
  except ValueError:
    print("Insufficient minority samples for SMOTETomek")
    return None

# COMMAND ----------

# MAGIC %md ##### Model Generation

# COMMAND ----------

counter = 0
print(f"Generating models for all solutions for type {model_type}")

for Solution in cust_50['LEVEL_CODE']:
  print(Solution)
  level = cust_50.loc[cust_50['LEVEL_CODE'] == Solution, ['LEVEL_NAME']].values.min()  
  
  counter = counter + 1
  
  if counter > 1000:
      break

  AD = DVs.filter((DVs.LEVEL_NAME == level) & (DVs.LEVEL_CODE == Solution)).select("PBUP_AC", "LEVEL_CODE", "LEVEL_NAME", "DV", "Period")\
  .join(masterdata,["PBUP_AC"], how='right')\
  .withColumn("LEVEL_CODE", lit(Solution))\
  .join(OpenYear,["PBUP_AC", "Period"], how='left')\
  .join(OpenQuarter,["PBUP_AC", "Period"], how='left')\
  .join(FirstWB,["PBUP_AC", "Period"], how='left')\
  .withColumn("random", rand())    

  OpenYear_IDs = OpenYear.select("PBUP_AC").dropDuplicates()
  AD = AD.join(OpenYear_IDs,["PBUP_AC"], how ='inner')

  if AD.count() == 0:
      continue

  print("Sampling data ...")

  AD = AD.filter((AD["DV"] == 1) |
          (((AD["PBPACTST"] == "E0004") | (AD["PBPACTST"] == "E0006") | (AD["PBPACTST"] == "E0007") | (AD["PBPACTST"] == "E0008")) & ((AD["PTARGACC"] == "C") | (AD["PTARGACC"] == "O") | (AD["PTARGACC"] == "")) & (AD["random"] < 0.15)) |\
          (((AD["PBPACTST"] == "E0004") | (AD["PBPACTST"] == "E0006") | (AD["PBPACTST"] == "E0007") | (AD["PBPACTST"] == "E0008")) & (AD["PTARGACC"] == "P") & (AD["random"] < 0.05)) |\
          (((AD["PBPACTST"] == "E0004") | (AD["PBPACTST"] == "E0006") | (AD["PBPACTST"] == "E0007") | (AD["PBPACTST"] == "E0008")) & ((AD["PTARGACC"] != "C") & (AD["PTARGACC"] != "O") & (AD["PTARGACC"] != "") & (AD["PTARGACC"] != "P")) & (AD["random"] < 0.25)) |\
          (((AD["PBPACTST"] != "E0004") & (AD["PBPACTST"] != "E0006") & (AD["PBPACTST"] != "E0007") & (AD["PBPACTST"] != "E0008")) & ((AD["PTARGACC"] == "C") | (AD["PTARGACC"] == "O") | (AD["PTARGACC"] == "")) & (AD["random"] < 0.075)) |\
          (((AD["PBPACTST"] != "E0004") & (AD["PBPACTST"] != "E0006") & (AD["PBPACTST"] != "E0007") & (AD["PBPACTST"] != "E0008")) & (AD["PTARGACC"] == "P") & (AD["random"] < 0.025)) |\
          (((AD["PBPACTST"] != "E0004") & (AD["PBPACTST"] != "E0006") & (AD["PBPACTST"] != "E0007") & (AD["PBPACTST"] != "E0008")) & ((AD["PTARGACC"] != "C") & (AD["PTARGACC"] != "O") & (AD["PTARGACC"] != "") & (AD["PTARGACC"] != "P")) & (AD["random"] < 0.125)))\
  .drop("PBPACTST")

  AD = AD.na.fill(0)
  AD_pd = AD.toPandas()

  #create binary dummies for categories
  AD_pd = pd.concat([AD_pd,
                 pd.get_dummies(AD_pd['PMASTERC'], prefix = 'PMASTERC_')],
                 axis = 1)
  AD_pd = pd.concat([AD_pd,
                 pd.get_dummies(AD_pd['PISLSGTM4'], prefix = 'PISLSGTM4_')],
                 axis = 1)
  AD_pd.drop(['random', 'PBUP_AC', 'LEVEL_CODE', 'Period', 'LEVEL_NAME', 'PMASTERC', 'PISLSGTM4', 'IND_CODE', 'PCRM_IMS', 'PBPOWNDFG', 'PTARGACC', 'PPBPINDI'], axis =1, inplace=True)

  X = AD_pd.drop("DV",axis=1)

  # Line 503 to 528 were added recently to add exclusions and were not tested. If there is any error. pls comment this code and check
    
  sa_hier = sa_flat_df.select("P1", "P2", "P3", "P4", "P5", "P6").drop_duplicates().toPandas()
  Nodes = sa_hier[(sa_hier.P1 == Solution) | (sa_hier.P2 == Solution) | (sa_hier.P3 == Solution) | (sa_hier.P4 == Solution) | (sa_hier.P5 == Solution) | (sa_hier.P6 == Solution)]

  P5 = Nodes[pd.isnull(Nodes['P4']) == False]
  P5 = P5['P4'].unique()
  P6 = Nodes[pd.isnull(Nodes['P5']) == False]
  P6 = P6['P5'].unique()
  print("Adding exclusions ....")
  exclusions = []

  for Node in P5: 
      exclusions.append("OpenOpps_12months_L5_" + Node)
      exclusions.append("OpenOpps_3months_L5_" + Node)
      exclusions.append("OpenOpps_12months_L5_" + Node + "1")
      exclusions.append("OpenOpps_3months_L5_" + Node + "1")

  for Node in P6: 
      exclusions.append("First_WB_Opp_L6_" + Node)

  exclusions.append("OpenOpps_12months_L5_PFOTHO1")
  exclusions.append("OpenOpps_3months_L5_PFOTHO1")
  exclusions.append("First_WB_Opp_L6_PFOTHO")

  columns = list(X)

  X = X.drop(columns=list(set(exclusions) & set(columns)))
  # y = AD_pd["DV"] 
  y = AD_pd.loc[:, 'DV'].values.astype("int") # Output/Dependent value

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0)
  
  print("Feature Selection in progress..")  
  corr_features = correlation(X_train, 0.60)
  X_train_net = X_train.drop(corr_features, axis=1)
  X_test_net = X_test.drop(corr_features, axis=1)

  if len(np.unique(y_train)) > 1:
    X_train_ns, y_train_ns = smotetomek(X_train_net, y_train)
    if X_train_ns is not None:
      print("Fitting Model")
      #Random Forest
      param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [8, 12, 16],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
      'class_weight': ['balanced', None]  # Consider class weights for imbalanced data
      }

      model = RandomForestClassifier(random_state=42)

      random_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='f1', n_iter=50)
      random_search.fit(X_train_ns, y_train_ns)  # Train with scaled and selected features

      # Get the best model and hyperparameters
      clf_final = random_search.best_estimator_ 
      clf_final.feature_names = list(X_train_ns.columns.values)

      ### write model
      _, path = tempfile.mkstemp()
      dump(clf_final, path)
      if model_type == "consulting":
        remote_path = os.path.join(adls_storage_path, f"{Solution}_CONS.sav")
      else:
        remote_path = os.path.join(adls_storage_path, f"{Solution}.sav") 
      print(f"Writing model file to {remote_path}")
      # client.put(path, remote_path) # Gen1
      with open(path, 'rb') as local_model_content:
        file_contents = local_model_content.read()

        file_client = client.get_file_client(remote_path)
        file_client.upload_data(data=file_contents, overwrite=True)

        os.remove(path)

      print(f"=> Calculating Accuracy for the model {Solution}")
      accuracy_calculate(clf_final, X_test_net, y_test, client, Solution, level, adls_storage_path)
      print(str(counter) + ': ' + Solution)
    else:
      continue
  else:
    print(f"Iteration skipped due to only one class present in y_train")

# COMMAND ----------

# MAGIC %md ### End of notebook
