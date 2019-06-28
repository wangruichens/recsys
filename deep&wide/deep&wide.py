# Auther        : wangruichen
# Date          : 2019-06-28
# Description   :
# Refers        :
# Returns       :

import tensorflow as tf
from tensorflow_estimator import estimator
import numpy as np
from time import time
import sys

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

FLAGS = tf.app.flags.FLAGS
# Model
tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout rate")
tf.app.flags.DEFINE_string("task_type", 'train', "Task type {train, infer, eval, export}")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_string("deep_layers", '100,100', "deep layers")
tf.app.flags.DEFINE_integer("cross_layers", 4, "cross layers")

# Dataset
tf.app.flags.DEFINE_string("train_path", '/home/wangrc/criteo_data/train/', "Data path")
tf.app.flags.DEFINE_integer("train_parts", 150, "Tfrecord counts")
tf.app.flags.DEFINE_integer("eval_parts", 5, "Eval tfrecord")
tf.app.flags.DEFINE_string("test_path", '/home/wangrc/criteo_data/test/', "Test path")
tf.app.flags.DEFINE_integer("test_parts", 15, "Tfrecord counts")

# Config
tf.app.flags.DEFINE_string("export_path", './export/', "Model export path")
tf.app.flags.DEFINE_integer("batch_size", 256, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 100, "Log_step_count_steps")
tf.app.flags.DEFINE_integer("save_checkpoints_steps", 2000, "save_checkpoints_steps")
tf.app.flags.DEFINE_integer("num_parallel", 8, "Number of batch size")
tf.app.flags.DEFINE_boolean("mirror", False, "Mirrored Strategy")

cont_feature = ['_c{0}'.format(i) for i in range(0, 14)]
cat_feature = ['_c{0}'.format(i) for i in range(14, 40)]

# Not setting default value for continuous feature. filled with mean.
feature_description = {k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in cont_feature}
feature_description.update({k: tf.FixedLenFeature(dtype=tf.string, shape=1, default_value='NULL') for k in cat_feature})


def build_feature_columns(embedding_size):
    cont_feature.remove('_c0')

    linear_feature_columns = []
    embedding_feature_columns = []

    # sorted(list(set(df.approxQuantile("a", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.01))))
    c1 = [0.0, 1.0, 2.0, 3.0, 5.0, 12.0]
    c2 = [0.0, 1.0, 2.0, 4.0, 10.0, 28.0, 76.0, 301.0]
    c3 = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0, 24.0, 54.0]
    c4 = [1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 13.0, 20.0]
    c5 = [20.0, 155.0, 1087.0, 1612.0, 2936.0, 5064.0, 8622.0, 16966.0, 39157.0]
    c6 = [3.0, 7.0, 13.0, 24.0, 36.0, 53.0, 85.0, 154.0, 411.0]
    c7 = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 17.0, 43.0]
    c8 = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 17.0, 25.0, 37.0]
    c9 = [4.0, 8.0, 16.0, 28.0, 41.0, 63.0, 109.0, 147.0, 321.0]
    c10 = [0.0, 1.0, 2.0]
    c11 = [0.0, 1.0, 2.0, 3.0, 4.0, 8.0]
    c12 = [0.0, 1.0, 2.0]
    c13 = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 22.0]
    buckets_cont = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]

    buckets_cat = [1460, 583, 10131226, 2202607, 305, 23, 12517, 633, 3, 93145, 5683, 8351592, 3194, 27, 14992, 5461305,
                   10, 5652, 2172, 3, 7046546, 17, 15, 286180, 104, 142571]

    buckets_cat = [1460, 583, 100000, 100000, 305, 23, 12517, 633, 3, 93145, 5683, 100000, 3194, 27, 14992, 100000,
                   10, 5652, 2172, 3, 100000, 17, 15, 100000, 104, 100000]

    for i, j in zip(cont_feature, buckets_cont):
        # Can not using eplison. if x=0, log (x+eplison) may cause nan loss.
        f_num = tf.feature_column.numeric_column(i, normalizer_fn=lambda x: tf.log(x + 1.0))
        f_bucket = tf.feature_column.bucketized_column(f_num, j)
        f_embedding = tf.feature_column.embedding_column(f_bucket, embedding_size)

        # TODO: With duplicated one-hot or not?
        # linear_feature_columns.append(tf.feature_column.indicator_column(f_bucket))
        linear_feature_columns.append(f_num)
        embedding_feature_columns.append(f_embedding)

    for i, j in zip(cat_feature, buckets_cat):
        f_cat = tf.feature_column.categorical_column_with_hash_bucket(key=i, hash_bucket_size=j)

        f_ind = tf.feature_column.indicator_column(f_cat)
        f_embedding = tf.feature_column.embedding_column(f_cat, embedding_size)

        # According to the paper, Sparse feature is used only as embeddings.
        # linear_feature_columns.append(f_ind)
        embedding_feature_columns.append(f_embedding)

    return linear_feature_columns, embedding_feature_columns

def _parse_examples(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features=feature_description)
    labels = features.pop('_c0')
    return features, labels


def input_fn(filenames, batch_size, num_epochs=-1, need_shuffle=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_examples, num_parallel_calls=FLAGS.num_parallel).batch(batch_size)
    if need_shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.prefetch(buffer_size=1000).repeat(num_epochs)
    return dataset

def main(_):
    data_dir = FLAGS.train_path
    data_files = []
    for i in range(FLAGS.train_parts):
        data_files.append(data_dir + 'part-r-{:0>5}'.format(i))

    train_files = data_files[:-FLAGS.eval_parts]
    eval_files = data_files[-FLAGS.eval_parts:]

    linear_feature_columns, embedding_feature_columns = build_feature_columns(FLAGS.embedding_size)

    distribute_strategy = None
    if FLAGS.mirror:
        distribute_strategy = tf.distribute.MirroredStrategy()
    config = estimator.RunConfig(
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=5,
        log_step_count_steps=FLAGS.log_steps,
        save_summary_steps=200,
        train_distribute=distribute_strategy,
        eval_distribute=distribute_strategy
    )

    model_params = {
        'linear_feature_columns': linear_feature_columns,
        'embedding_feature_columns': embedding_feature_columns,
        'embedding_size': FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "dropout": FLAGS.dropout,
        "deep_layers": FLAGS.deep_layers
    }

    dcn = estimator.LinearClassifier(
        model_dir='./model/',
        feature_columns=linear_feature_columns,
        config=config)

    if FLAGS.task_type == 'train':
        train_spec = estimator.TrainSpec(input_fn=lambda: input_fn(
            train_files,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            need_shuffle=True))
        eval_spec = estimator.EvalSpec(input_fn=lambda: input_fn(
            eval_files,
            num_epochs=-1,
            batch_size=FLAGS.batch_size,
            need_shuffle=True), steps=200, start_delay_secs=1, throttle_secs=5)
        estimator.train_and_evaluate(dcn, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        dcn.evaluate(input_fn=lambda: input_fn(eval_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=200)


if __name__ == '__main__':
    # tf.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
