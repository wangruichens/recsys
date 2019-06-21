# Auther        : wangruichen
# Date          : 2019-06-21
# Description   :
# Refers        :
# Returns       :

from tensorflow_estimator import estimator
import tensorflow as tf
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

cont_feature = ['_c{0}'.format(i) for i in range(0, 14)]
cat_feature = ['_c{0}'.format(i) for i in range(14, 40)]

# Not setting default value for continuous feature. filled with mean.
feature_description = {k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in cont_feature}
feature_description.update({k: tf.FixedLenFeature(dtype=tf.string, shape=1, default_value='NULL') for k in cat_feature})


def build_feature(embedding_size):
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

    for i, j in zip(cont_feature, buckets_cont):
        f_num = tf.feature_column.numeric_column(i,
                                                 default_value=0,
                                                 normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
        f_bucket = tf.feature_column.bucketized_column(f_num, j)
        f_embedding = tf.feature_column.embedding_column(f_bucket)

        # TODO: With duplicated one-hot or not?
        # linear_feature.append(tf.feature_column.indicator_column(f_bucket))

        linear_feature_columns.append(f_num)
        embedding_feature_columns.append(f_embedding)

    for i, j in zip(cat_feature, buckets_cat):
        f_cat = tf.feature_column.categorical_column_with_hash_bucket(key=i, buckets=j)

        f_ind = tf.feature_column.indicator_column(f_cat)
        f_embedding = tf.feature_column.embedding_column(f_cat, embedding_size)

        linear_feature_columns.append(f_ind)
        embedding_feature_columns.append(f_embedding)

    return linear_feature_columns, embedding_feature_columns


def _parse_examples(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features=feature_description)
    labels = features.pop('_c0')
    return features, labels


def input_fn(filenames, batch_size, num_epochs=-1, need_shuffle=False):
    """
            The function should construct and return one of the following:
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
            `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
            of string feature name to `Tensor` and `labels` is a `Tensor` or a
            dictionary of string label name to `Tensor`.

            Both `features` and `labels` are consumed by `model_fn`. They should satisfy the expectation
            of `model_fn` from inputs.
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_examples, num_parallel_calls=4).batch(batch_size)
    if need_shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.prefetch(buffer_size=100).repeat(num_epochs)
    return dataset


def model_fn(features, labels, mode, params):

    layers = list(map(int, params["deep_layers"].split(',')))

    linear_net = tf.feature_column.input_layer(features, params['linear_feature_columns'])
    embedding_net = tf.feature_column.input_layer(features, params['embedding_feature_columns'])

    with tf.name_scope('linear_net'):
        linear_y = tf.layers.dense(linear_net, 1, activation=tf.nn.relu)
    with tf.name_scope('cin_net'):



    with tf.name_scope('dnn_net'):
        dnn_net = tf.reshape(embedding_net,shape=[-1, len(params['embedding_feature_columns']) * params['embedding_size']])
        for i in range(len(layers)):
            dnn_net = tf.layers.dense(dnn_net, i, activation=tf.nn.relu)

            dnn_net = tf.layers.batch_normalization(dnn_net, training=(mode == estimator.ModeKeys.TRAIN))
            dnn_net = tf.layers.dropout(dnn_net, rate=params['dropout'], training=(mode == estimator.ModeKeys.TRAIN))
        dnn_y = tf.layers.dense(dnn_net, 1, activation=tf.nn.relu)

    return NotImplementedError


def main(_):
    # tf.enable_eager_execution()
    # dataset = input_fn(['part-r-00000'], 1)
    # for raw_record in dataset.take(1):
    #     print(repr(raw_record))

    build_feature(32)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
