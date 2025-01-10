# Databricks notebook source
!pip install --upgrade scikit-learn imbalanced-learn
!pip install -U imbalanced-learn

# COMMAND ----------

!pip install smote-variants

# COMMAND ----------

# MAGIC %md #### Initialize

# COMMAND ----------

dbutils.widgets.dropdown("bw_name", "BWP", ["BWP", "BWT", "BWV"], "Select Relevant BW")
dbutils.widgets.text("bw_username", "C5368226", "BW Username")
dbutils.widgets.text("bw_scope_name", "C5368226", "BW Scope Name")
dbutils.widgets.text("bw_scope_key", "C5368226_BWP", "BW Scope Key Name")
dbutils.widgets.dropdown("type", "generic", ["generic", "consulting"], "Model Type")

# process_new = Manually Apply Generic & process_existing = Apply Generic
dbutils.widgets.dropdown("mode", "process_new", ["process_new", "process_existing"], "Mode")

# COMMAND ----------

# MAGIC %md ##### Load utils notebook

# COMMAND ----------

# MAGIC %run ./pm_utils

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
from datetime import datetime, timedelta

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

from azure.datalake.store import core, lib
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity  import ClientSecretCredential

import os
import tempfile
import time
from sqlalchemy import create_engine

from dataclasses import dataclass
from functools import reduce

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from pyspark.sql import DataFrame
#from pyspark.sql.functions import *
from pyspark.sql.functions import year, month, add_months, quarter, lit, concat, datediff, to_date, rand, Column, array, arrays_zip, explode, col, trim

spark = init_spark_with_adls_cred(spark)
client = get_corp_datalake_client()

# bw_credentials = get_bw_credentials(
#   bw_instance = 'BWP',
#   username = 'C5368226',
#   password = 'Melmaruvathur%13569')
bw_credentials = get_bw_credentials(
  bw_instance = dbutils.widgets.get("bw_name"),
  username = dbutils.widgets.get("bw_username"),
  password = dbutils.secrets.get(
    scope = dbutils.widgets.get("bw_scope_name"), key = dbutils.widgets.get("bw_scope_key"))
)

run_mode = dbutils.widgets.get("mode")
print(f"=> Run mode is {run_mode}")

model_type = dbutils.widgets.get("type")
print(f"Model type is {model_type}")

table_identifier = ""
if model_type == "consulting":
  table_identifier = "_CONS"

OUTPUT_SCHEMA_NAME = "EXAN_PRIVATE"
OUTPUT_NAMESPACE_NAME = "corp.exan.tables"
STAGING_OUTPUT_TABLE_NAME = f"PM{table_identifier}_OUTPUT_STAGING_SA_BP"
HISTORY_OUTPUT_TABLE_NAME = f"PM{table_identifier}_OUTPUT_HISTORY_SA_BP"
FULL_OUTPUT_TABLE_FORMAT = '"{}"."{}::{}.{}"'
FULL_STAGING_TABLE_NAME = FULL_OUTPUT_TABLE_FORMAT.format(OUTPUT_SCHEMA_NAME, OUTPUT_NAMESPACE_NAME, STAGING_OUTPUT_TABLE_NAME, STAGING_OUTPUT_TABLE_NAME)
FULL_HISTORY_TABLE_NAME = FULL_OUTPUT_TABLE_FORMAT.format(OUTPUT_SCHEMA_NAME, OUTPUT_NAMESPACE_NAME, HISTORY_OUTPUT_TABLE_NAME, HISTORY_OUTPUT_TABLE_NAME)
print(f"  - The output tables are {STAGING_OUTPUT_TABLE_NAME} and {HISTORY_OUTPUT_TABLE_NAME}")

base_adls_storage_path = "store/PropensityModels/2024/BP/Aug"

models_adls_path = f"{base_adls_storage_path}/models/{model_type}/"
print(f"  - The model path on ADLS is {models_adls_path}")

current_date = datetime.today()
end_date = current_date + timedelta(days=2)
end_date = end_date.strftime("%Y%m%d")
scores_staging_adls_path = f"{base_adls_storage_path}/scores/{model_type}/staging/{end_date}/"
print(f"  - The staging score path on ADLS is {scores_staging_adls_path}")

scores_history_adls_path = f"{base_adls_storage_path}/scores/{model_type}/history/"
print(f"  - The history score path on ADLS is {scores_history_adls_path}")

SOLUTION_AREA_HIERARCHY_YEAR = "2024"
print(f"Solution area hierarchy used is: {SOLUTION_AREA_HIERARCHY_YEAR}")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### DATA PREPARATION

# COMMAND ----------

# MAGIC %md ##### Read masterdata

# COMMAND ----------

# Read Masterdata
if model_type == "consulting":
  print(f"Model type is {model_type} and the master data will be joined with CL_CORP_MD_CUSTOMER")
  sql = "(select a.PMASTERC, a.IND_CODE, a.PCRM_IMS, a.PISLSGTM4, a.PBPOWNDFG, a.PTARGACC, a.PPBPINDI, a.PBPACTST, a.PBUP_AC \
          from \"_SYS_BIC\".\"corp.exan.views/CL_EXAN_PM_MASTERDATA\" as a \
          left join \"_SYS_BIC\".\"corp.mada.sac.dimensions/CL_CORP_MD_CUSTOMER\" as b \
            on a.\"PBUP_AC\" = b.\"_BIC_PBUP_AC\" \
          left join \"_SYS_BIC\".\"corp.mada.sac.dimensions/CL_CORP_MD_BP_ROLES\" as c \
            on a.PBUP_AC = c.PARTNER \
        where PPBPINDI not in ('A', 'L', 'U', 'X', 'Y') \
        and a.PBUP_AC <> '' \
        and a.PISLSGTM4 not in ('01', '') \
        and a.PBPNOTREL <> 'X' \
        and a.PBPXDELE <> 'X' \
        and a.PBPXBLCK <> 'X' \
        and c.RLTYP <> 'BUP004' \
        group by a.PMASTERC, a.IND_CODE, a.PCRM_IMS, a.PISLSGTM4, a.PBPOWNDFG, a.PTARGACC, a.PPBPINDI, a.PBPACTST, a.PBUP_AC, a.PBUP_AC)"
else:
  print(f"Model type is {model_type} and the master data will be joined with CL_CORP_MD_PBUP_AC")
  
  sql = "(select a.PMASTERC, a.IND_CODE, a.PCRM_IMS, a.PISLSGTM4, a.PBPOWNDFG, a.PTARGACC, a.PPBPINDI, a.PBPACTST, a.PBUP_AC, c.PSEGMENT \
          from \"_SYS_BIC\".\"corp.exan.views/CL_EXAN_PM_MASTERDATA\" as a \
          left join \"_SYS_BIC\".\"corp.mada.sac.dimensions/CL_CORP_MD_PBUP_AC\" as c \
                on a.PBUP_AC = c.PBUP_AC \
          left join \"_SYS_BIC\".\"corp.mada.sac.dimensions/CL_CORP_MD_BP_ROLES\" as b \
                on a.PBUP_AC = b.PARTNER \
        where a.PPBPINDI not in ('A', 'L', 'U', 'X', 'Y') \
        and a.PBUP_AC <> '' \
        and a.PISLSGTM4 not in ('01', '') \
        and a.PBPNOTREL <> 'X' \
        and a.PBPXDELE <> 'X' \
        and a.PBPXBLCK <> 'X' \
        and b.RLTYP <> 'BUP004' \
        group by a.PMASTERC, a.IND_CODE, a.PCRM_IMS, a.PISLSGTM4, a.PBPOWNDFG, a.PTARGACC, a.PPBPINDI, a.PBPACTST, a.PBUP_AC, c.PSEGMENT)"
 
masterdata = run_sql_with_spark(bw_credentials, sql)
masterdata_pd = masterdata.toPandas() # Used only for Generic models

# COMMAND ----------

# MAGIC %md ##### Read sa and pipeline data

# COMMAND ----------

sa_flat_df = get_solution_area_data(SOLUTION_AREA_HIERARCHY_YEAR)
pipeline = (get_pipeline_with_sa(sa_flat_df)
            .select('IC_ACCOUNT_ID', 'IC_OBJECT_ID', 'IC_STATUS', 'IC_STATUS_SINCE', 'CA_STATUS_SINCE_DATE', 'PBUP_AC', 'LEVEL_CODE', 'LEVEL_NAME')
            .withColumn("CA_STATUS_SINCE_DATE", col("CA_STATUS_SINCE_DATE").cast("date"))
            .filter(trim(col("LEVEL_CODE")) != '')
           )
pipeline.cache()

# COMMAND ----------

# Get consulting Opps
sql = '''(select IC_OBJECT_ID
    FROM "_SYS_BIC"."corp.exan.views/CL_GSPI_IC_PIPELINE_ALL" 
    where ITM_TYPE in ('CONS', 'SUPP', 'EDU', 'ZCOS', 'SERV')
    group by IC_OBJECT_ID)'''
 
consulting = run_sql_with_spark(bw_credentials, sql)
 
pipeline_cons = pipeline.join(consulting, on = ["IC_OBJECT_ID"], how = 'inner')

# COMMAND ----------

# MAGIC %md ##### Open Opportunities last Quater

# COMMAND ----------

i = 1 # quater number

OpenQuarter_trans = (pipeline
    .filter(pipeline["LEVEL_NAME"] == 'LEVEL6')
    .filter((pipeline["IC_STATUS"] == 'E0001') | (pipeline["IC_STATUS"] == 'E0007'))
    .withColumn("LEVEL_CODE", concat(lit('OpenOpps_3months_L6_'), pipeline.LEVEL_CODE))
    .groupBy("PBUP_AC", "LEVEL_CODE")
    .count()
    .withColumn("counter", lit(1)))

OpenQuarter = OpenQuarter_trans.groupBy(['PBUP_AC']).pivot('LEVEL_CODE').sum('count')

# COMMAND ----------

# MAGIC %md ##### Open Opportunities last year

# COMMAND ----------

OpenYear_trans = (pipeline
    .filter(pipeline["LEVEL_NAME"] == 'LEVEL6')
    .filter((pipeline["IC_STATUS"] == 'E0001') | (pipeline["IC_STATUS"] == 'E0007'))
    .withColumn("LEVEL_CODE", concat(lit('OpenOpps_12months_L6_'), pipeline.LEVEL_CODE))
    .filter(pipeline["IC_STATUS_SINCE"] >= get_date_in_past(-3 * i - 12, -3 * (i + 1))) # Year Start Date (Go back 1 year)
    .filter(pipeline["IC_STATUS_SINCE"] < get_date_in_past(-3 * i)) # Year End Date
    .groupBy("PBUP_AC", "LEVEL_CODE")
    .count()
    .withColumn("counter", lit(1)))

OpenYear = OpenYear_trans.groupBy(['PBUP_AC']).pivot('LEVEL_CODE').sum('count')

# COMMAND ----------

# MAGIC %md ##### Duration since Won/Booked

# COMMAND ----------

# duration since first open opportunity
FirstWB_trans = (pipeline
.filter(pipeline["LEVEL_NAME"] == 'LEVEL6')
.filter((pipeline["IC_STATUS"] == 'E0001') | (pipeline["IC_STATUS"] == 'E0007'))
.withColumn("LEVEL_CODE", concat(lit('First_WB_Opp_L6_'), pipeline.LEVEL_CODE))
.filter(pipeline["IC_STATUS_SINCE"] > 19000101)
.filter(pipeline["IC_STATUS_SINCE"] < get_date_in_past(-3 * (i)))
.groupBy("PBUP_AC", "LEVEL_CODE")
.agg({'CA_STATUS_SINCE_DATE': 'min'})
.withColumnRenamed("min(CA_STATUS_SINCE_DATE)", "CA_STATUS_SINCE_DATE"))

FirstWB_trans = (FirstWB_trans
                 .withColumn(
                   "diff",
                   datediff(
                     to_date(lit(datetime.strptime(str(get_date_in_past(-3 * (i))), "%Y%m%d")), 'yyyyMMdd'),
                     FirstWB_trans["CA_STATUS_SINCE_DATE"]))
                )

FirstWB = FirstWB_trans.groupBy(['PBUP_AC']).pivot('LEVEL_CODE').mean('diff')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Application dataset

# COMMAND ----------

masterdata.cache().count()
OpenQuarter.cache().count()
OpenYear.cache().count()
FirstWB.cache().count()

ApplyDS = masterdata\
.join(OpenQuarter,["PBUP_AC"], how='left')\
.join(OpenYear,["PBUP_AC"], how='left')\
.join(FirstWB,["PBUP_AC"], how='left')

ApplyDS = ApplyDS.na.fill(0)

# COMMAND ----------

ApplyDS_pd = ApplyDS.toPandas()
spark.catalog.clearCache()

# COMMAND ----------

# creating dummy variables
ApplyDS_pd = pd.concat([ApplyDS_pd,
                   pd.get_dummies(ApplyDS_pd['PMASTERC'], prefix = 'PMASTERC_')],
                   axis = 1)

ApplyDS_pd = pd.concat([ApplyDS_pd,
                   pd.get_dummies(ApplyDS_pd['PISLSGTM4'], prefix = 'PISLSGTM4_')],
                   axis = 1)

ApplyDS_pd = ApplyDS_pd.reset_index()

# COMMAND ----------

# extract first occurence of won / booked pipeline for exclusion flag
if model_type == "consulting":
    print(f"Model type is consulting, hence using the pipeline_cons object")
    SolutionPipe = (pipeline_cons
        .filter((pipeline_cons["IC_STATUS"] == 'E0001') | (pipeline_cons["IC_STATUS"] == 'E0007'))
        .groupBy("PBUP_AC", "LEVEL_CODE").agg({'IC_STATUS_SINCE': 'min'}).toPandas())
else:
    print(f"Model type is generic, hence using the pipeline object")
    SolutionPipe = (pipeline
        .filter((pipeline["IC_STATUS"] == 'E0001') | (pipeline["IC_STATUS"] == 'E0007'))
        .groupBy("PBUP_AC", "LEVEL_CODE").agg({'IC_STATUS_SINCE': 'min'}).toPandas())

SolutionPipe.columns = ['PBUP_AC', 'LEVEL_CODE', 'IC_STATUS_SINCE']

# COMMAND ----------

# MAGIC %md
# MAGIC ## apply to all models in "EXAN_PRIVATE"."corp.exan.tables::PM_OUTPUT_STAGING.PM_OUTPUT_STAGING"

# COMMAND ----------

output_db_credentials = bw_credentials

# COMMAND ----------

def get_existing_model_name():
  # Default sort order is ascending (so we are selecting solutions sorted by date)
  sql = f'''
    (SELECT TOP 50 SOLUTION 
     FROM {FULL_STAGING_TABLE_NAME}
     WHERE SOLUTION LIKE '%.{SOLUTION_AREA_HIERARCHY_YEAR}'
     GROUP BY SOLUTION , DATE 
     ORDER BY DATE ASC)
  '''
  return run_sql_with_spark(output_db_credentials, sql).collect()

# COMMAND ----------

def get_new_model_names(models_adls_path):
  print(f"Getting models list from adls using path {models_adls_path}")
  #solution_files = client.ls(models_adls_path) # Gen1
  solution_files = client.get_paths(models_adls_path, recursive=False) # Gen2

  adls_solution_list = []
  for file_object in solution_files:
    base_path, file_name = os.path.split(file_object.name)
    if ".sav" in file_name:
      if model_type == "consulting":
        solution_name = file_name.replace("_CONS.sav", "")
      else:
        solution_name = file_name.replace(".sav", "")
      adls_solution_list.append(solution_name)
  
  # sql = f'''
  #   (SELECT "SOLUTION"
  #   FROM {FULL_STAGING_TABLE_NAME}
  #   WHERE "DATE" > '20201028' and SOLUTION like '%.{SOLUTION_AREA_HIERARCHY_YEAR}'
  #   GROUP BY "SOLUTION", "DATE"
  #   ORDER BY "DATE" DESC)
  # '''
  # bw_solutions_list_raw = run_sql_with_spark(output_db_credentials, sql).collect()
  # bw_solutions_list = []
  # for bw_solution in bw_solutions_list_raw:
  #   bw_solutions_list.append(bw_solution.SOLUTION)
  # return set(adls_solution_list) - set(bw_solutions_list)

  return adls_solution_list

# COMMAND ----------

if run_mode == "process_new":
  SolutionsList = get_new_model_names(models_adls_path)
  where = ", ".join([f"'{solution}'" for solution in SolutionsList])
else:
  # For existing inference, we select the oldest 50 models and create a where clause
  solution_list_row = get_existing_model_name()
  SolutionsList = []
  for solution in solution_list_row:
    SolutionsList.append(solution['SOLUTION'])

  where = ", ".join([f"'{solution}'" for solution in SolutionsList])
  print(SolutionsList)

print(f"For model type: {model_type} / run mode: {run_mode}, total solutions found are {len(SolutionsList)}")
print(where)

# COMMAND ----------

# extract existing scores
block_reading_scores = ""
if run_mode == "process_new":
  print(f"Blocking the read of existing scores since we are in Manually Apply Generic/Consulting mode")
  block_reading_scores = "1=2 and"

sql = f'(select * from {FULL_STAGING_TABLE_NAME} where {block_reading_scores} SOLUTION in ({where}))'
existing_scores = run_sql_with_spark(output_db_credentials, sql)
existing_scores.cache().count()

# COMMAND ----------

# extract historic conversions
sql = f'(select PBUP_AC as "PBUP_AC", DATE as ConversionDate, SOLUTION from {FULL_HISTORY_TABLE_NAME} WHERE SOLUTION LIKE \'%.{SOLUTION_AREA_HIERARCHY_YEAR}\')'

historic_conversion = run_sql_with_spark(output_db_credentials, sql)
historic_conversion.cache().count()

# COMMAND ----------

def write_parquet_to_adls(df, solution_name, adls_base_path):
  _, local_path = tempfile.mkstemp()
  df.to_parquet(local_path, index=False)

  adls_base_path_with_date = os.path.join(adls_base_path, str(datetime.today().strftime("%Y%m%d")))
  adls_full_path = os.path.join(adls_base_path_with_date, solution_name)

  # client.put(local_path, adls_full_path) # Gen1
  with open(local_path, 'rb') as local_path_content:
    file_contents = local_path_content.read()

    file_client = client.get_file_client(adls_full_path)
    file_client.upload_data(data=file_contents, overwrite=True)

# COMMAND ----------

import joblib
def load_model_from_adls(models_adls_path, solution_name, model_type):
  fd, path = tempfile.mkstemp()
  if model_type == "consulting":
    remote_path = os.path.join(models_adls_path, f"{solution_name}_CONS.sav")
  else:
    remote_path = os.path.join(models_adls_path, f"{solution_name}.sav")

  print(f"  - Loading model {remote_path}")

  #client.get(remote_path, path) # Gen1
  with open(path, "wb") as file_stream:
    file_client = client.get_file_client(remote_path)
    file_client.download_file().readinto(file_stream)
  
  loaded_model = joblib.load(path)
  os.close(fd)
  return loaded_model

# COMMAND ----------

import urllib.parse
password = urllib.parse.quote('Melmaruvathur%13569')

# COMMAND ----------

print("Starting inference")
DATE = date.today().strftime("%Y%m%d")
step = time.time()

url_extract = f"hana://{output_db_credentials.username}:{password}@{output_db_credentials.host}:{output_db_credentials.port}"
engine = create_engine(url_extract, connect_args={'encrypt': 'True', 'sslValidateCertificate':'False'})
print("Database object initialized")

counter = 0

IDs = ApplyDS_pd['PBUP_AC']
ApplyDS_pd = ApplyDS_pd.fillna(0)
print("Nulls filled with zero")

for Solution in SolutionsList:
  counter = counter + 1
  
  print(f"=> Run: {str(counter)} for {Solution}")
  
  loaded_model = load_model_from_adls(models_adls_path, Solution, model_type)
  
  # Apply model in 500k batches into a pandas dataframe    
  i = 1
  output_temp = pd.DataFrame()
  while (i - 1) * 500000 < IDs.count():
    print(f"  - Prediction iteration: {str(i)}")
    sample = pd.DataFrame(columns=loaded_model.feature_names_in_)
    sample = pd.concat([sample, ApplyDS_pd.iloc[(i - 1) * 500000:i * 500000, :]], sort=False)
    sample = sample.drop(columns=list(set(sample.columns).difference(loaded_model.feature_names_in_)))
    sample = sample.fillna(0)

    exclusions = ["DV"]
    columns = list(ApplyDS_pd)
    sample = sample.drop(columns=list(set(exclusions) & set(columns)))
    y_pred = loaded_model.predict_proba(sample)
    output_temp = pd.concat([output_temp, pd.DataFrame(y_pred)], axis=0)
    print("  - %s minutes ---" % ((time.time() - step) / 60))
    step = time.time()
           
    i = i + 1
  if (len(output_temp.columns) < 2):
    print('!!! No propensities > 0 can be retrieved !!!')
    continue
  
  # create output file
  output_temp = output_temp.reset_index()
  output_temp['Solution'] = Solution
  output_temp = pd.concat([pd.DataFrame(IDs.iloc[0:]), output_temp], axis=1)
  output_temp.columns = ['PBUP_AC', 'tobedropped', 'prob_0', 'prob_1', 'Solution']
  output_temp = output_temp.drop(columns=['tobedropped'])
  output_temp['DATE'] = date.today().strftime("%Y%m%d")
  output_temp = output_temp.dropna(subset=['PBUP_AC'])

  if model_type == "generic":
    # join masterdata to deprioritize missing Internal Account Segment and missing Internal Market Segment

    output_temp = pd.merge(output_temp, masterdata_pd[['PISLSGTM4', 'PBUP_AC', 'PSEGMENT']], on='PBUP_AC', how='inner')
    output_temp.prob_1 = np.where(((output_temp['PISLSGTM4'] == '') | 
                                  (output_temp['PISLSGTM4'] == '01') | 
                                  (output_temp['PSEGMENT'] == 'N/R') | 
                                  (output_temp['PSEGMENT'] == 'N/A')), output_temp['prob_1'] / 100, output_temp['prob_1'])
  
  # extract existing scores
  print('  - Read existing model data from BW')
  existing = existing_scores.filter(existing_scores["SOLUTION"] == Solution).toPandas()
  existing.columns = map(str.upper, existing.columns)
  # check whether the model has been scored before today - if yes take Opps since last score else since today
  if existing["DATE"].count() > 1:
      LastRefresh = existing['DATE'].agg(['max'])
      LastRefresh = LastRefresh[0]
  else:
      LastRefresh = datetime.today().strftime("%Y%m%d")
  
  # extract historic conversions
  history = historic_conversion.filter(historic_conversion["SOLUTION"] == Solution).toPandas()
  history.columns = map(str.upper, history.columns)
  
  # extract won / booked pipe after last refresh
  SolutionPipeNew = SolutionPipe[(SolutionPipe.LEVEL_CODE == Solution) & (SolutionPipe.IC_STATUS_SINCE >= LastRefresh)]

  # new opportunities
  PredictedOpps = pd.merge(existing, SolutionPipeNew.PBUP_AC, on='PBUP_AC', how='inner')
  PredictedOpps = PredictedOpps.drop_duplicates()
  PredictedOpps = pd.merge(PredictedOpps, history, on= ['PBUP_AC', 'SOLUTION'], how='left')
  PredictedOpps = PredictedOpps.drop_duplicates()
  PredictedOpps = PredictedOpps[PredictedOpps["CONVERSIONDATE"].isnull()]
  PredictedOpps = PredictedOpps.drop(columns=['CONVERSIONDATE'])

  if model_type == "consulting":
    PredictedOpps = PredictedOpps[['PBUP_AC', 'DATE', 'SCORE', 'SOLUTION', 'EXCLUDE', 'LOGICALPRODUCT', 'PERCENTILE', 'DECILE', 'FLAG1', 'FLAG2', 'FLAG3', 'FLAG4', 'FLAG5']]
  else:
    PredictedOpps = PredictedOpps[['PBUP_AC', 'DATE', 'SCORE', 'SOLUTION', 'EXCLUDE', 'LOGICALPRODUCT', 'PERCENTILE', 'DECILE']]
  print("  - %s minutes ---" % ((time.time() - step) / 60))
  step = time.time()

  # write predicted opp history to BW
  print(f"  - Write HISTORY scores to BW ({HISTORY_OUTPUT_TABLE_NAME}) and ADLS ({scores_history_adls_path})")
  engine = create_engine(url_extract, connect_args={'encrypt': 'True', 'sslValidateCertificate':'False'}) 
  PredictedOpps.to_sql(f'{OUTPUT_NAMESPACE_NAME}::{HISTORY_OUTPUT_TABLE_NAME}.{HISTORY_OUTPUT_TABLE_NAME}', schema=OUTPUT_SCHEMA_NAME, con=engine, if_exists='append', index=False)
 
  write_parquet_to_adls(PredictedOpps, Solution, scores_history_adls_path)

  # write scores to BW'
  output_temp = output_temp[['PBUP_AC', 'DATE', 'prob_1', 'Solution']]
  output_temp.columns = ['PBUP_AC', 'DATE', 'SCORE', 'SOLUTION']
  SolutionPipeNew['EXCLUDE'] = 1
  output_temp = pd.merge(output_temp, SolutionPipeNew[['PBUP_AC', 'EXCLUDE']], on='PBUP_AC', how='left')
  output_temp = output_temp.drop_duplicates()
  print("------")
  print(SolutionPipeNew)
  output_temp['LOGICALPRODUCT'] = np.where((output_temp.SOLUTION.str.match('LPR')) | (output_temp.SOLUTION.str.match('LSV')), 1, 0)
  output_temp['PERCENTILE'] = pd.qcut(output_temp['SCORE'].rank(method='first'), 100, labels=False)
  print("------")
  print(output_temp['PERCENTILE'])
  print("------")
  output_temp['PERCENTILE'] = abs(output_temp['PERCENTILE'] + 1 - 101)
  # Red flag: Why output_temp['SCORE'] is not used here. Percentile doesn't make sense
  output_temp['DECILE'] = pd.qcut(output_temp['PERCENTILE'].rank(method='first'), 10, labels=False)
  output_temp['DECILE'] = abs(output_temp['DECILE'] + 1 - 11)

  if model_type == "consulting":
    print(f"  - Settings flags for consulting models output")
    # flag "Sweet Spot" = all records with a score higher than the minimum of percentile 1
    minimum = output_temp.loc[output_temp['PERCENTILE'] == 1, 'SCORE'].min()
    output_temp['FLAG1'] = np.where((output_temp.SCORE > minimum), 1, 0)
    output_temp['FLAG2'] = np.nan
    output_temp['FLAG3'] = np.nan
    output_temp['FLAG4'] = np.nan
    output_temp['FLAG5'] = np.nan
  
  # delete old entries from the table
  print(f"  - Write STAGING scores to BW ({STAGING_OUTPUT_TABLE_NAME}) and ADLS ({scores_staging_adls_path})")
  engine = engine.connect()
  engine.execute(f"delete from {FULL_STAGING_TABLE_NAME} where SOLUTION = '{Solution}'")

  # write refreshed scores into BW
  engine = create_engine(url_extract, connect_args={'encrypt': 'True', 'sslValidateCertificate':'False'})
  output_temp.to_sql(f'{OUTPUT_NAMESPACE_NAME}::{STAGING_OUTPUT_TABLE_NAME}.{STAGING_OUTPUT_TABLE_NAME}', schema=OUTPUT_SCHEMA_NAME, con=engine, if_exists='append', index=False)
  
  write_parquet_to_adls(output_temp, Solution, scores_staging_adls_path)

  print("  - %s minutes ---" % ((time.time() - step) / 60))
  step = time.time()

# COMMAND ----------


