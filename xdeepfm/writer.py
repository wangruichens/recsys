# Auther        : wangruichen
# Date          : 2019-06-21
# Description   : write csv to hive
# Refers        :
# Returns       :

from pyspark.sql import SparkSession
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", default='hdfs://cluster/user/wangrc/criteo/test.txt')
    parser.add_argument("--db", default='mlg')
    parser.add_argument("-t","--table", default='')
    args = parser.parse_args()
    return args

def df_to_hive(spark, df, table_name):
    tmp_table_name = "tmp_" + table_name
    df.registerTempTable(tmp_table_name)
    delete_sql = "drop table if exists " + table_name
    create_sql = "create table " + table_name + " as select * from " + tmp_table_name
    spark.sql(delete_sql)
    spark.sql(create_sql)



def main(args):
    ss = SparkSession.builder \
        .appName("writer") \
        .enableHiveSupport() \
        .getOrCreate()
    ss.sql(f'use {args.db}')

    print("writing '{0}' into hive...".format(args.path))
    df = ss.read.format("csv").option("header", "false").option("inferSchema", "true").option("sep", "\t").load(args.path)
    df_to_hive(ss,df,args.table)
    print("done ...")

if __name__ == '__main__':
    args = parse_args()
    main(args)
