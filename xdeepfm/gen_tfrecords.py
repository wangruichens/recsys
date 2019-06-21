import argparse
from pyspark.sql import SparkSession


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default='mlg')
    args = parser.parse_args()
    return args


def main(args):
    ss = SparkSession.builder \
        .appName("write_tfrecords") \
        .enableHiveSupport() \
        .getOrCreate()
    ss.sql(f'use {args.db}')

    train_df = ss.sql("select * from mlg.wangrc_criteo_train")
    train_df.repartition(100).write.format("tfrecords") \
        .mode("overwrite").option("recordType", "Example") \
        .save('/user/wangrc/criteo_data/train/')

    test_df = ss.sql("select * from mlg.wangrc_criteo_test")
    test_df.repartition(10).write.format("tfrecords") \
        .mode("overwrite").option("recordType", "Example") \
        .save('/user/wangrc/criteo_data/test/')


if __name__ == '__main__':
    args = parse_args()
    main(args)
