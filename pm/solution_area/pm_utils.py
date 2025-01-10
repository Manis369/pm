# Databricks notebook source
from azure.datalake.store import core, lib
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity  import ClientSecretCredential

import os
import tempfile

from dataclasses import dataclass
from functools import reduce

import pandas as pd
import numpy as np
from joblib import *

from sklearn.ensemble import RandomForestClassifier        
from sklearn.model_selection import GridSearchCV

from pyspark.sql import DataFrame
#from pyspark.sql.functions import *
from pyspark.sql.functions import year, month, add_months, quarter, lit, concat, datediff, to_date, rand, Column, array, arrays_zip, explode, col, trim
from datetime import date
from dateutil.relativedelta import relativedelta

# COMMAND ----------

def install_cluster_dependencies():
  dbutils.library.installPyPI("azure-datalake-store", version="0.0.48")
  #dbutils.library.installPyPI("scikit-learn", version="0.23.1")
  dbutils.library.installPyPI("scikit-learn", version="0.24.2")
  dbutils.library.installPyPI("sqlalchemy", version="1.4.46")
  dbutils.library.installPyPI("sqlalchemy-hana", version="0.5.0")
  dbutils.library.installPyPI("pandas", version="1.5.3")

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class BWCredentials:
  connection_string: str
  instance: str
  username: str
  password: str
  host: str
  port: str
  driver: str = "com.sap.db.jdbc.Driver"
                                                                           

def get_bw_credentials(bw_instance: str, username: str, password: str) -> BWCredentials:
  bw_ip_dict = {
    "BWP": "10.67.58.220",
    "BWT": "10.67.58.210",
    "BWV": "10.67.59.29"
  }
  bw_default_port = "30241"
  
  return BWCredentials(
    connection_string = f"jdbc:sap://{bw_ip_dict.get(bw_instance)}:{bw_default_port}",
    instance = bw_instance,
    username = username,
    password = password,
    host = bw_ip_dict.get(bw_instance),
    port = bw_default_port
  )

# COMMAND ----------

def init_spark_with_gen1_adls_cred(spark):
  spark.conf.set("fs.adl.oauth2.access.token.provider.type", "ClientCredential")
  spark.conf.set("fs.adl.oauth2.client.id", dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_ID"))
  spark.conf.set("fs.adl.oauth2.credential", dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_SECRET"))
  spark.conf.set("fs.adl.oauth2.refresh.url", "https://login.microsoftonline.com/42f7676c-f455-423c-82f6-dc2d99791af7/oauth2/token")
  return spark


def init_spark_with_adls_cred(spark):
  storage_account = "coredatalakeprodint"
  spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
  spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
  spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net", dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_ID"))
  spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net", dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_SECRET"))
  spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net", "https://login.microsoftonline.com/42f7676c-f455-423c-82f6-dc2d99791af7/oauth2/token")
  return spark

# COMMAND ----------

def merge_dictionary(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def run_sql_with_spark(credentials: BWCredentials, sql_command: str = None, custom_options: dict = {}) -> DataFrame:
  default_options = {
    "driver": credentials.driver,
    "url": credentials.connection_string,
    "user": credentials.username,
    "password": credentials.password,
    "encrypt": True,
    "validateCertificate": False
  }

  if sql_command:
     default_options["query"] = sql_command

  #print(default_options)
  options = {}
  if custom_options:
    options = merge_dictionary(default_options, custom_options)
  else:
    options = default_options

  #print(options)
  return (spark
          .read
          .format("jdbc")
          .options(**options)
#           .option("driver", credentials.driver)
#           .option("url", credentials.connection_string)
#           .option("user", credentials.username)
#           .option("password", credentials.password)
#           .option("query", sql_command)
#           .option("fetchsize", fetch_size)
          #.option("dbtable", sql_command)
          .load())

# COMMAND ----------

def get_corp_gen1_datalake_client():
  token = lib.auth(tenant_id='42f7676c-f455-423c-82f6-dc2d99791af7',
                 client_secret=dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_SECRET"), 
                 client_id=dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_ID"), 
                 resource="https://datalake.azure.net/")
  return core.AzureDLFileSystem(token, store_name="datalakemdeea")
  

def get_corp_datalake_client():
  storage_account = "coredatalakeprodint"
  credential = ClientSecretCredential(tenant_id="42f7676c-f455-423c-82f6-dc2d99791af7",
                                      client_id=dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_ID"),
                                      client_secret=dbutils.secrets.get(scope = "PROD_EXAN_SCOPE", key = "SPN_113_PROD_EXAN_SECRET"))
  service_client = DataLakeServiceClient(account_url=f"https://{storage_account}.dfs.core.windows.net", credential=credential)

  file_system_client = service_client.get_file_system_client(file_system="exan")
  return file_system_client

# COMMAND ----------

def get_datetime_in_past(months_to_go_back: int, quater_override: int = None, add_empty_time: bool = True) -> int:
  current_date = date.today()
  back_date = (current_date + relativedelta(months=months_to_go_back))
  
  year = back_date.year
  
  if quater_override:
    quater_back_date = (current_date + relativedelta(months=quater_override))
    quarter = (quater_back_date.month-1)//3+1
  else:
    quarter = (back_date.month-1)//3+1

  first_day_of_quarter = 1
  first_month_of_quarter = 3 * quarter - 2
  
  date_yyyymmdd = (year * 10000) + (first_month_of_quarter * 100) + (first_day_of_quarter)

  if add_empty_time:
    #print(year, quarter, first_month_of_quarter, date_yyyymmdd, date_time)
    return date_yyyymmdd * 1000000
  else:
    #print(year, quarter, first_month_of_quarter, date_yyyymmdd)
    return date_yyyymmdd

def get_date_in_past(months_to_go_back: int, quater_override: int = None) -> Column:
  return get_datetime_in_past(months_to_go_back, quater_override, False)

# COMMAND ----------

def read_master_data():
  # Read Masterdata  - exclude records w/o Planning Entity, w/o Internal Sales Segment and with ERP ID
  # PBPNOTREL = Not released/related
  # PBPXDELE = Deleted
  # PBPXBLCK = Black listed
  sql = "(select a.PMASTERC, a.IND_CODE, a.PCRM_IMS, a.PISLSGTM4, a.PBPOWNDFG, a.PTARGACC, a.PPBPINDI, a.PBPACTST, a.PPLNENTY_FINAL, a.PBUP_AC \
          from \"_SYS_BIC\".\"corp.exan.views/CL_EXAN_PM_MASTERDATA\" as a \
          left join \"_SYS_BIC\".\"corp.mada.sac.dimensions/CL_CORP_MD_CUSTOMER\" as b \
            on a.\"PBUP_AC\" = b.\"_BIC_PBUP_AC\" \
          left join \"_SYS_BIC\".\"corp.mada.sac.dimensions/CL_CORP_MD_BP_ROLES\" as c \
            on a.PBUP_AC = c.PARTNER \
        where a.PPBPINDI not in ('A', 'L', 'U', 'X', 'Y') \
        and a.PBUP_AC <> '' \
        and a.PISLSGTM4 not in ('01', '') \
        and a.PBPNOTREL <> 'X' \
        and a.PBPXDELE <> 'X' \
        and a.PBPXBLCK <> 'X'  \
        and c.RLTYP <> 'BUP004' \
        group by a.PMASTERC, a.IND_CODE, a.PCRM_IMS, a.PISLSGTM4, a.PBPOWNDFG, a.PTARGACC, a.PPBPINDI, a.PBPACTST, a.PPLNENTY_FINAL, a.PBUP_AC)"
  
  return (run_sql_with_spark(bw_credentials, sql)
           .drop(
                 "CH_ON",
                 "PBPXDELE",
                 "PBPXBLCK",
                 "PBPNOTREL",
                 "PISLSGTM4_T",
                 "PPLNENTY",
                 "CUSTOMER",
                 "PBPACTST_T",
                "CH_ON_DATE"))

# COMMAND ----------

def get_solution_area_data(hierarchy_year) -> DataFrame:
    # Get Solution Area Flat Table
  solution_area_flat_sql = f'''
      (SELECT
       LTRIM("MATERIAL",'0') AS MATERIAL_ID,
       "MATERIAL_DESC",
       "MATERIAL_LEVEL_01_CODE" AS "P1",
       "MATERIAL_LEVEL_01_DESC" AS "P1_TEXT",
       "MATERIAL_LEVEL_02_CODE" AS "P2",
       "MATERIAL_LEVEL_02_DESC" AS "P2_TEXT",
       "MATERIAL_LEVEL_03_CODE" AS "P3",
       "MATERIAL_LEVEL_03_DESC" AS "P3_TEXT",
       "MATERIAL_LEVEL_04_CODE" AS "P4",
       "MATERIAL_LEVEL_04_DESC" AS "P4_TEXT",
       "MATERIAL_LEVEL_05_CODE" AS "P5",
       "MATERIAL_LEVEL_05_DESC" AS "P5_TEXT",
       "MATERIAL_LEVEL_06_CODE" AS "P6",
       "MATERIAL_LEVEL_06_DESC" AS "P6_TEXT"
  FROM "_SYS_BIC"."corp.mada.sac.dimensions/CL_CORP_MADA_HIER_MATERIAL_FLAT"('PLACEHOLDER' = ('$$P_INFOOBJECT$$',
       '0MATERIAL'),
       'PLACEHOLDER' = ('$$P_MATERIAL_HIERARCHY$$',
       'SOLAREA.{hierarchy_year}'))
  )
  '''

  return run_sql_with_spark(bw_credentials, solution_area_flat_sql)


# COMMAND ----------

def get_pipeline_with_sa(sa_flat_df: DataFrame) -> DataFrame:
  # Get Pipeline Data
  pipeline_data_sql = '''
  (SELECT
    "IC_ACCOUNT_ID",
    "IC_OBJECT_ID",
    "IC_PHASE",
    "SP_PHASE_SINCE",
    "IC_PHASE_TXT",
    "IC_STATUS_SINCE",
    "IC_STATUS",
    "CA_STATUS_SINCE_DATE",
    "CREATED_AT",
    "PRODUCT_ID",
    "ITM_TYPE",
    TO_NVARCHAR("PRODUCT_ID") AS "PRODUCT_ID_NVARCHAR",
    TO_DATE("CREATED_AT") AS "CREATED_AT_DATE"
  FROM "_SYS_BIC"."corp.gspi.datamart.predictive.core/CL_GSPI_IC_PIPELINE_ALL"('PLACEHOLDER' = ('$$IP_CURRENCY$$', 'EUR'))
  WHERE CA_STATUS_SINCE_DATE > ADD_MONTHS(CURRENT_DATE, -40))
  '''

  pipeline_df = run_sql_with_spark(bw_credentials, pipeline_data_sql)

  sa_flat_df.cache()

  sa_flat_grouped_df = (sa_flat_df
     .select("MATERIAL_ID",
       array("P1", "P2", "P3", "P4", "P5", "P6").alias("P_CODES"),
       array("P1_TEXT", "P2_TEXT", "P3_TEXT", "P4_TEXT", "P5_TEXT", "P6_TEXT").alias("P_TEXT"),
       array(lit("LEVEL1"), lit("LEVEL2"), lit("LEVEL3"), lit("LEVEL4"), lit("LEVEL5"), lit("LEVEL6")).alias("P_LEVEL")
     )
  )

  sa_flat_exploded_df =(sa_flat_grouped_df
   .withColumn("zipped_cols", arrays_zip("P_CODES", "P_TEXT", "P_LEVEL"))
   .withColumn("zipped_cols", explode("zipped_cols"))
   .select("MATERIAL_ID",
           col("zipped_cols.P_CODES").alias("LEVEL_CODE"),
           col("zipped_cols.P_TEXT").alias("LEVEL_TEXT"),
           col("zipped_cols.P_LEVEL").alias("LEVEL_NAME")
          ))

  # Get PBUP_AC and PPLNENTY_FINAL
  pbup_ac_df = '''
      (SELECT
        "PBUP_AC"
        FROM "_SYS_BIC"."corp.mada.sac.dimensions/CL_CORP_MD_PBUP_AC"
      )
  '''

  pbup_ac_df = run_sql_with_spark(bw_credentials, pbup_ac_df)


  # Join Pipeline data with solution Area
  pipeline_sa_join_df = (
    pipeline_df
      .join(sa_flat_exploded_df,
            on = pipeline_df.PRODUCT_ID_NVARCHAR == sa_flat_df.MATERIAL_ID,
            how='inner')
      .drop("PRODUCT_ID_NVARCHAR")
  )

  # Join Pipeline x SA with PBUP
  pipeline_sa_pbup_join_df = (
    pipeline_sa_join_df
      .join(pbup_ac_df,
            on = pipeline_sa_join_df.IC_ACCOUNT_ID == pbup_ac_df.PBUP_AC,
            how = 'inner')
      
  )
  return pipeline_sa_pbup_join_df

# COMMAND ----------

def get_pipeline_with_pf() -> DataFrame:
  """
  Just to test if the function and view for PF are exactly equal or not.
  The count is exact equal and function works correctly.
  """
  pipeline_data_sql = '''
  (SELECT
    "IC_ACCOUNT_ID",
    "IC_OBJECT_ID",
    "IC_PHASE",
    "SP_PHASE_SINCE",
    "IC_PHASE_TXT",
    "IC_STATUS_SINCE",
    "IC_STATUS",
    "CA_STATUS_SINCE_DATE",
    "CREATED_AT",
    "PRODUCT_ID",
    "ITM_TYPE",
    TO_NVARCHAR("PRODUCT_ID") AS "PRODUCT_ID_NVARCHAR",
    TO_DATE("CREATED_AT") AS "CREATED_AT_DATE"
  FROM "_SYS_BIC"."corp.gspi.datamart.predictive.core/CL_GSPI_IC_PIPELINE_ALL"('PLACEHOLDER' = ('$$IP_CURRENCY$$', 'EUR'))
  WHERE CA_STATUS_SINCE_DATE > ADD_MONTHS(CURRENT_DATE, -40)) 
  '''

  pipeline_df = run_sql_with_spark(bw_credentials, pipeline_data_sql)

  
  pf_flat_sql = '''
      (SELECT
       "PORT_STRUC_KEY",
       "PORT_STRUC_TEXT",
       "P1",
       "P1_TEXT",
       "P2",
       "P2_TEXT",
       "P3",
       "P3_TEXT",
       "P4",
       "P4_TEXT",
       "P5",
       "P5_TEXT",
       "P6",
       "P6_TEXT"
  FROM "_SYS_BIC"."corp.mada.fac.dimensions/CL_CORP_MD_HIER_PORTFOLIO_CY_FLAT_ALL"
  )
  '''

  
  pf_flat_df = run_sql_with_spark(bw_credentials, pf_flat_sql)
  pf_flat_df.cache()

  pf_flat_grouped_df = (pf_flat_df
     .select("PORT_STRUC_KEY",
       array("P1", "P2", "P3", "P4", "P5", "P6").alias("P_CODES"),
       array("P1_TEXT", "P2_TEXT", "P3_TEXT", "P4_TEXT", "P5_TEXT", "P6_TEXT").alias("P_TEXT"),
       array(lit("LEVEL1"), lit("LEVEL2"), lit("LEVEL3"), lit("LEVEL4"), lit("LEVEL5"), lit("LEVEL6")).alias("P_LEVEL")
     )
  )

  pf_flat_exploded_df =(pf_flat_grouped_df
   .withColumn("zipped_cols", arrays_zip("P_CODES", "P_TEXT", "P_LEVEL"))
   .withColumn("zipped_cols", explode("zipped_cols"))
   .select("PORT_STRUC_KEY",
           col("zipped_cols.P_CODES").alias("LEVEL_CODE"),
           col("zipped_cols.P_TEXT").alias("LEVEL_TEXT"),
           col("zipped_cols.P_LEVEL").alias("LEVEL_NAME")
          ))

  # Get PBUP_AC and PPLNENTY_FINAL
  pbup_ac_df = '''
      (SELECT
        "PBUP_AC",
        "PPLNENTY",
        CASE WHEN "PPLNENTY" = '' THEN TRIM("PBUP_AC") ELSE TRIM("PPLNENTY") END AS "PPLNENTY_FINAL"
        FROM "_SYS_BIC"."corp.mada.sac.dimensions/CL_CORP_MD_PBUP_AC"
      )
  '''

  pbup_ac_df = run_sql_with_spark(bw_credentials, pbup_ac_df)


  # Join Pipeline data with solution Area
  pipeline_sa_join_df = (
    pipeline_df
      .join(pf_flat_exploded_df,
            on = pipeline_df.PRODUCT_ID_NVARCHAR == pf_flat_df.PORT_STRUC_KEY,
            how='inner')
      .drop("PRODUCT_ID_NVARCHAR")
  )

  # Join Pipeline x SA with PBUP
  pipeline_sa_pbup_join_df = (
    pipeline_sa_join_df
      .join(pbup_ac_df,
            on = pipeline_sa_join_df.IC_ACCOUNT_ID == pbup_ac_df.PBUP_AC,
            how = 'inner')
      .drop(pbup_ac_df.PBUP_AC)
      .drop(pbup_ac_df.PPLNENTY)
  )
  return pipeline_sa_pbup_join_df

# COMMAND ----------


