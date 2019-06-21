import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default='mlg')
    args = parser.parse_args()
    return args

def preprocess(df):
    cont_col = ['_c{0}'.format(i) for i in range(0, 14)]
    for i in cont_col:
        df = df.withColumn(i, df[i].cast("float"))
    # Continuous columns fill null with mean
    imputer=Imputer(inputCols=cont_col,outputCols=cont_col).setStrategy('mean')

    return imputer.fit(df).transform(df)


def main(args):
    ss = SparkSession.builder \
        .appName("write_tfrecords") \
        .enableHiveSupport() \
        .getOrCreate()
    ss.sql(f'use {args.db}')

    train_df = ss.sql("select * from mlg.wangrc_criteo_train")
    train_df=preprocess(train_df)
    train_df.repartition(100).write.format("tfrecords") \
        .mode("overwrite").option("recordType", "Example") \
        .save('/user/wangrc/criteo_data/train/')


    test_df = ss.sql("select * from mlg.wangrc_criteo_test")
    test_df=preprocess(test_df)
    test_df.repartition(10).write.format("tfrecords") \
        .mode("overwrite").option("recordType", "Example") \
        .save('/user/wangrc/criteo_data/test/')


if __name__ == '__main__':
    args = parse_args()
    main(args)
