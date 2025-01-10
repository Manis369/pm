# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("bw_name", "BWP", ["BWP", "BWT", "BWV"], "Select Relevant BW")
dbutils.widgets.text("bw_username", "C5368226", "BW Username")
dbutils.widgets.text("bw_scope_name", "C5368226", "BW Scope Name")
dbutils.widgets.text("bw_scope_key", "C5368226_BWP", "BW Scope Key Name")
dbutils.widgets.dropdown("use_small_dataset", "No", ["Yes", "No"] , "Use Small Dataset")


# COMMAND ----------

# MAGIC %run ./pm_utils

# COMMAND ----------

install_cluster_dependencies()
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

import pyhdb
from sqlalchemy import create_engine
from pyspark.sql.functions import *
from pyspark.sql.window import Window

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

STAGING_TABLE_NAME = "PM_OUTPUT_STAGING_SA_BP"
AGGREGATED_TABLE_NAME = "PM_OUTPUT_AGGREGATED_SA_BP"
base_adls_storage_path = "store/PropensityModels/2024/BP/Aug"
model_type = 'generic'
scores_staging_adls_path = f"{base_adls_storage_path}/scores/{model_type}/staging"
print(scores_staging_adls_path)
agg_scores_staging_adls_path = f"{base_adls_storage_path}/scores/{model_type}/staging/agg"
print(agg_scores_staging_adls_path)
use_small_dataset = False
if dbutils.widgets.get("use_small_dataset") == "Yes":
  use_small_dataset = True

print(f"Use small dataset is set to {use_small_dataset}")

# COMMAND ----------

def filter_full_hierarchies(filtered_sa_df):
  """
  Solution Area can have many columns and the hierarchy
  is defined in the table. Sometimes, some levels are empty.
  This function, filters only rows which contain all levels.
  """
  desired_columns = ["P1", "P2", "P3", "P4", "P5", "P6"]

  for c in desired_columns:
    filtered_sa_df = filtered_sa_df.filter(col(c) != '')

  sa_6levels_df = filtered_sa_df.select(*desired_columns).drop_duplicates()
  return sa_6levels_df


def combine_level5_and_6_in_one_column(sa_6levels_df):
  """
  As generic propensity model work on level 5 and 6 and
  we need a mapping between 5/6 to level 2 and 3. Hence,
  we move level 5 and 6 column in a single column.
  """
  sa_level56_combined_df = (sa_6levels_df
    .select("P1", "P2", "P3", "P4", array("P5", "P6").alias("level5and6"))
    .withColumn("level5or6", explode("level5and6"))
    .select("P1", "P2", "P3", "P4", "level5or6"))
  return sa_level56_combined_df

SOLUTION_AREA_HIERARCHY_YEAR = "2024"
print(f"Solution area hierarchy used is: {SOLUTION_AREA_HIERARCHY_YEAR}")

sa_df = get_solution_area_data(SOLUTION_AREA_HIERARCHY_YEAR)
full_hierarchies_df = filter_full_hierarchies(sa_df)
sa_level56_combined_df = combine_level5_and_6_in_one_column(full_hierarchies_df)

display(sa_df)
display(sa_level56_combined_df)

# COMMAND ----------

query_suffix = ""
if use_small_dataset:
  query_suffix = "LIMIT 1000"

sql = f'''(SELECT * 
           FROM  "EXAN_PRIVATE"."corp.exan.tables::{STAGING_TABLE_NAME}.{STAGING_TABLE_NAME}"
           WHERE LOGICALPRODUCT = 1 and
                 SOLUTION like '%.{SOLUTION_AREA_HIERARCHY_YEAR}'
           {query_suffix}) '''
print(sql)

custom_options = {
    "partitionColumn": "PERCENTILE",
    "lowerBound": 0,
    "upperBound": 100,
    "numPartitions": 100,
    "fetchsize": 10000,
    "dbtable": sql
}

scores = run_sql_with_spark(bw_credentials, None, custom_options)
scores.cache()

# COMMAND ----------

# current_date = datetime.today()
# d_day = current_date.strftime("%Y%m%d")

# path = f"abfss://exan@coredatalakeprodint.dfs.core.windows.net/store/PropensityModels/2024/BP/Aug/scores/generic/staging/{d_day}/"

# def read_all_parquet_files(path):
#   custom_options = {
#       "partitionColumn": "PERCENTILE",
#       "lowerBound": 0,
#       "upperBound": 100,
#       "numPartitions": 100,
#       "fetchsize": 10000
#   }

#   df = spark.read.format("parquet").option("recursiveFileLookup", True).options(**custom_options).load(path)
#   return df

# predictions = read_all_parquet_files(path)
# predictions.show()

# COMMAND ----------

if use_small_dataset:
  print("Joining against small random sample")
  scores_with_sa_df = (
     scores
      .join(broadcast(sa_level56_combined_df), scores.SOLUTION == sa_level56_combined_df.level5or6)
    ).orderBy(rand()).limit(1000)
else:
  print("Joining against full dataset")
  scores_with_sa_df = (
     scores
      .join(broadcast(sa_level56_combined_df), scores.SOLUTION == sa_level56_combined_df.level5or6)
    )

display(scores_with_sa_df)

# COMMAND ----------

# Testing by converting to pandas and qcut
avg_score_df = (scores_with_sa_df
 .select("PBUP_AC", "DATE", "SCORE", "SOLUTION", "EXCLUDE", "LOGICALPRODUCT", "PERCENTILE", "DECILE", "P3")
 .groupBy("P3", "PBUP_AC").agg({"SCORE": "max", "DATE": "max"}) # Aggregated table has a unique key on scenario, pplnenty and solution and multiple dates can repeat
 .withColumnRenamed("max(DATE)", "DATE")
 .withColumnRenamed("max(SCORE)", "SCORE")
 .groupBy(col("P3"), col("DATE"), col("PBUP_AC"))  # TODO: This aggregation is redundant since previous aggregation already leaves unique rows for PPLNENTY_FINAL, DATE and P3
 .agg(max(col("SCORE")).alias("SCORE"))
 .withColumn("LOGICALPRODUCT", lit(0))
 .withColumn("SCENARIO", lit("Solution Area Level 3"))
 .withColumn("EXCLUDE", lit(None).cast('int'))
 .withColumnRenamed("P3", "SOLUTION")
)

avg_score_pd_df = avg_score_df.toPandas()

# COMMAND ----------

import urllib.parse
password = urllib.parse.quote('Melmaruvathur%13569')

# COMMAND ----------

print("Deleting old entries from database")

url_extract = f"hana://{bw_credentials.username}:{password}@{bw_credentials.host}:{bw_credentials.port}"
engine = create_engine(url_extract, connect_args={'encrypt': 'True', 'sslValidateCertificate':'False'})
engine = engine.connect()
delete_query = f'delete from "EXAN_PRIVATE"."corp.exan.tables::{AGGREGATED_TABLE_NAME}.{AGGREGATED_TABLE_NAME}"'
print(delete_query)
engine.execute(delete_query)

# COMMAND ----------

# Loop over all solutions
all_solutions = avg_score_pd_df.SOLUTION.drop_duplicates()
print(f"Total solutions are {len(all_solutions)}")

counter = 0
for Solution in all_solutions:
  print(f"{str(counter)} => [{Solution}]: Calculating percentile and decile")
  avg_score_per_solution = avg_score_pd_df[avg_score_pd_df.SOLUTION == Solution]

  avg_score_per_solution['SCORE'] = pd.to_numeric(avg_score_per_solution['SCORE'])
  avg_score_per_solution['PERCENTILE'] = pd.qcut(avg_score_per_solution['SCORE'].rank(method='first'), 100, labels=False)
  avg_score_per_solution['PERCENTILE'] = (avg_score_per_solution['PERCENTILE'] + 1 - 101)*-1
  avg_score_per_solution['DECILE'] = pd.qcut(avg_score_per_solution['SCORE'].rank(method='first'), 10, labels=False)
  avg_score_per_solution['DECILE'] = (avg_score_per_solution['DECILE'] + 1 - 11)*-1

  # Select the columns
  avg_score_per_solution = avg_score_per_solution[["SCENARIO", "PBUP_AC", "DATE", "SCORE", "SOLUTION", "EXCLUDE", "LOGICALPRODUCT", "PERCENTILE", "DECILE"]]

  print(f"{str(counter)} => [{Solution}]: Writing to database")
  avg_score_per_solution.to_sql(f'corp.exan.tables::{AGGREGATED_TABLE_NAME}.{AGGREGATED_TABLE_NAME}', schema="EXAN_PRIVATE", con=engine, if_exists='append', index=False)
  print(f"{str(counter)} => [{Solution}]: Finished writing")

  counter = counter + 1

# COMMAND ----------

sql_query = """(SELECT * 
                FROM "EXAN_PRIVATE"."corp.exan.tables::PM_OUTPUT_AGGREGATED_SA_BP.PM_OUTPUT_AGGREGATED_SA_BP")"""

custom_options = {
    "fetchsize": 10000,
    "query": sql_query
}

pm_df = run_sql_with_spark(bw_credentials, None, custom_options)
pm_df.cache()

# COMMAND ----------

agg_df = pm_df.groupBy("PBUP_AC") \
    .agg(F.round(F.avg("PERCENTILE"), 0).alias("AVG_PERCENTILE")) \
    .withColumn("SOI_CALC_DATE", F.date_format(F.current_date(), 'yyyyMMdd'))

# Step 3: Define a window for NTILE over AVG_PERCENTILE and PPLNENTY_FINAL
window_spec = Window.orderBy(F.col("AVG_PERCENTILE").asc(), F.col("PBUP_AC").asc())

# Step 4: Apply NTILE to split AVG_PERCENTILE into 100 buckets
agg_df = agg_df.withColumn("AVG_PERCENTILE_BUCKET", F.ntile(100).over(window_spec))

# Step 5: Apply conditional logic to create the SOI_CLASS column
agg_df = agg_df.withColumn(
    "SOI_CLASS",
    F.when((F.col("AVG_PERCENTILE_BUCKET") > 0) & (F.col("AVG_PERCENTILE_BUCKET") <= 10), "1 Top High")
     .when((F.col("AVG_PERCENTILE_BUCKET") > 10) & (F.col("AVG_PERCENTILE_BUCKET") <= 20), "2 High")
     .when((F.col("AVG_PERCENTILE_BUCKET") > 20) & (F.col("AVG_PERCENTILE_BUCKET") <= 40), "3 Medium High")
     .when((F.col("AVG_PERCENTILE_BUCKET") > 40) & (F.col("AVG_PERCENTILE_BUCKET") <= 60), "4 Medium Low")
     .when((F.col("AVG_PERCENTILE_BUCKET") > 60) & (F.col("AVG_PERCENTILE_BUCKET") <= 100), "5 Low")
)

# Step 6: Select the required columns
final_df = agg_df.select(
    "SOI_CALC_DATE",
    "PBUP_AC",
    F.col("AVG_PERCENTILE").alias("SOI_RAW_SCORE"),
    "SOI_CLASS"
)

# Step 7: Display the result
final_df.show(truncate=False)

# COMMAND ----------

sql_query = """(SELECT *
                FROM "corp.mada.sac.dimensions::CL_CORP_MD_PBUP_AC")"""

custom_options = {
    "fetchsize": 10000,
    "query": sql_query
}

md_empty_pe_mapped_to_bp_df = run_sql_with_spark(bw_credentials, None, custom_options)
display(md_empty_pe_mapped_to_bp_df)

md_empty_pe_mapped_to_bp_df = md_empty_pe_mapped_to_bp_df \
    .select(
        F.col("PBUP_AC"),
        F.when(F.col("PPLNENTY") == "", F.col("PBUP_AC")).otherwise(F.col("PPLNENTY")).alias("PPLNENTY")
    ).distinct()

md_empty_pe_mapped_to_bp_df.show(truncate=False)

# COMMAND ----------

sql_query = """(SELECT *
                FROM "corp.mada.sac.dimensions::CL_CORP_MADA_HIER_MATERIAL_FLAT"('PLACEHOLDER' = ('$$P_INFOOBJECT$$','0MATERIAL'),
        'PLACEHOLDER' = ('$$P_MATERIAL_HIERARCHY$$','SOLAREA.2024')))"""

custom_options = {
    "fetchsize": 10000,
    "query": sql_query
}

sa_hierarchy = run_sql_with_spark(bw_credentials, None, custom_options)
display(sa_hierarchy)
sa_hierarchy = sa_hierarchy \
    .filter(F.col("MATERIAL_LEVEL_03_CODE") != "") \
    .select("MATERIAL_LEVEL_03_CODE", "MATERIAL_LEVEL_03_DESC") \
    .distinct() \
    .orderBy("MATERIAL_LEVEL_03_CODE")

# Show the filtered and sorted data
sa_hierarchy.show(truncate=False)

# COMMAND ----------

sql_query = """(SELECT * 
                FROM "EXAN_PRIVATE"."corp.exan.tables::PM_OUTPUT_AGGREGATED_SA_BP.PM_OUTPUT_AGGREGATED_SA_BP")"""

custom_options = {
    "fetchsize": 10000,
    "query": sql_query
}

pm_aggregated_df = run_sql_with_spark(bw_credentials, None, custom_options)
pm_aggregated_df.cache()
pm_aggregated_data_with_lvl3_desc_df = pm_aggregated_df.alias("PM") \
    .join(sa_hierarchy.alias("SA_HIERARCHY"), 
          F.col("PM.SOLUTION") == F.col("SA_HIERARCHY.MATERIAL_LEVEL_03_CODE"), 
          "inner") \
    .select(
        F.col("PM.PBUP_AC").alias("BP_ID"),
        F.col("PM.PERCENTILE").alias("PERCENTILE"),
        F.col("PM.SOLUTION").alias("MODEL_ID"),
        F.col("PM.SCORE").alias("RAW_SCORE"),
        F.col("PM.DECILE").alias("DECILE"),
        F.col("PM.DATE").alias("PM_SCORE_DATE"),
        F.col("SA_HIERARCHY.MATERIAL_LEVEL_03_DESC").alias("MODEL_NAME")
    )

# Step 4: Display the final result
pm_aggregated_data_with_lvl3_desc_df.show(truncate=False)

# COMMAND ----------

pm_aggregated_data_with_lvl3_desc_with_bp_df = pm_aggregated_data_with_lvl3_desc_df.alias("PM") \
    .join(md_empty_pe_mapped_to_bp_df.alias("PE_BP"),
          F.col("PM.BP_ID") == F.col("PE_BP.PBUP_AC"),
          "left") \
    .select(F.col("PM.*"), F.col("PE_BP.PPLNENTY").alias("PE_ID"))

# COMMAND ----------

soi_df = final_df \
    .select(
        F.col("PBUP_AC"),
        F.col("SOI_CALC_DATE"),
        F.col("SOI_RAW_SCORE"),
        F.col("SOI_CLASS")
    )
pm_agg_soi_v2_df = pm_aggregated_data_with_lvl3_desc_with_bp_df.alias("PM") \
    .join(soi_df.alias("SOI"),
          F.col("PM.BP_ID") == F.col("SOI.PBUP_AC"), 
          "inner")

# COMMAND ----------

soi_bp_aggregated = pm_agg_soi_v2_df \
    .select(
        F.col("PE_ID"),
        F.col("BP_ID"),
        F.col("MODEL_ID"),
        F.col("MODEL_NAME"),
        F.col("PERCENTILE"),
        F.col("SOI_CLASS"),
        F.col("SOI_CALC_DATE")
    )

soi_bp_aggregated = soi_bp_aggregated.withColumnRenamed('SOI_CALC_DATE', 'REPORT_DATE') \
                                     .withColumnRenamed('BP_ID', 'CRM_ACCOUNT_ID')
soi_bp_aggregated = soi_bp_aggregated.drop('PE_ID')

# COMMAND ----------

soi_bp_aggregated = soi_bp_aggregated.select(
    'REPORT_DATE',
    'CRM_ACCOUNT_ID',
    'SOI_CLASS',
    'MODEL_NAME',
    'MODEL_ID',
    'PERCENTILE'   
)

# COMMAND ----------

ENV = 'prod'
def set_spark_config_to_access_adls_gen2(env):

    """
    set the spark config to access adls
    Input:
        env: str, staging/production
    """

    if env == "test":
        scope = "TEST_EXTR_SCOPE"
        client_id_key = "SPN_114_TEST_EXTR_ID"
        secret_key = "SPN_114_TEST_EXTR_SECRET"

        if env == "prod":
            scope = "PROD_EXTR_SCOPE"
            client_id_key = "SPN_16_PROD_EXTR_ID"
            secret_key = "SPN_16_PROD_EXTR_SECRET"
            # client_id_key = "SPN_113_PROD_EXAN_ID"
            # secret_key = "SPN_113_PROD_EXAN_SECRET"

        tenant_id = "42f7676c-f455-423c-82f6-dc2d99791af7"

        spark.conf.set(
            "fs.azure.account.auth.type.coredatalaketestint.dfs.core.windows.net",
            "OAuth",
        )
        spark.conf.set(
            "fs.azure.account.oauth.provider.type.coredatalaketestint.dfs.core.windows.net",
            "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
        )
        spark.conf.set(
            "fs.azure.account.oauth2.client.id.coredatalaketestint.dfs.core.windows.net",
            dbutils.secrets.get(scope=scope, key=client_id_key),
        )
        spark.conf.set(
            "fs.azure.account.oauth2.client.secret.coredatalaketestint.dfs.core.windows.net",
            dbutils.secrets.get(scope=scope, key=secret_key),
        )
        spark.conf.set(
            "fs.azure.account.oauth2.client.endpoint.coredatalaketestint.dfs.core.windows.net",
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
        )

set_spark_config_to_access_adls_gen2(ENV)

# COMMAND ----------

OUT_PATH = f"abfss://extr@coredatalaketestint.dfs.core.windows.net/data/store/PropensityModels/data/bp-soi"
soi_bp_aggregated.coalesce(1).write.mode("overwrite").parquet(OUT_PATH)

# COMMAND ----------

# MAGIC %md ## End of Notebook

# COMMAND ----------


