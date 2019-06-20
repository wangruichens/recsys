#!/bin/bash
source /etc/profile
spark-submit \
--master yarn \
--conf spark.network.timeout=600 \
--conf spark.sql.shuffle.partitions=10 \
--conf spark.executor.memoryOverhead=2048 \
--conf spark.driver.memoryOverhead=2048 \
--executor-cores 4 \
--num-executors 4 \
--executor-memory 4g \
--driver-memory 20g \
gen_tfrecords.py