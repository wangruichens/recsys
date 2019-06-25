# Auther        : wangruichen
# Date          : 2019-06-21
# Description   :
# Refers        :
# Returns       :

from tensorflow_estimator import estimator
from time import time
import tensorflow as tf
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout rate")
tf.app.flags.DEFINE_string("task_type", 'train', "Task type {train, infer, eval, export}")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs")
tf.app.flags.DEFINE_string("deep_layers", '100,100', "deep layers")
tf.app.flags.DEFINE_string("cross_layers", '10,5,5', "cross layers")

tf.app.flags.DEFINE_string("train_path", '/home/wangrc/criteo_data/train/', "Data path")
tf.app.flags.DEFINE_integer("train_parts", 150, "Tfrecord counts")
tf.app.flags.DEFINE_integer("eval_parts", 5, "Eval tfrecord")

tf.app.flags.DEFINE_string("test_path", '/home/wangrc/criteo_data/test/', "Test path")
tf.app.flags.DEFINE_integer("test_parts", 15, "Tfrecord counts")

tf.app.flags.DEFINE_string("export_path", './export/', "Model export path")
tf.app.flags.DEFINE_integer("batch_size", 256, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 50, "Log_step_count_steps")
tf.app.flags.DEFINE_integer("save_checkpoints_steps", 500, "save_checkpoints_steps")
tf.app.flags.DEFINE_boolean("mirror", True, "Mirrored Strategy")

cont_feature = ['_c{0}'.format(i) for i in range(0, 14)]
cat_feature = ['_c{0}'.format(i) for i in range(14, 40)]

# Not setting default value for continuous feature. filled with mean.
feature_description = {k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in cont_feature}
feature_description.update({k: tf.FixedLenFeature(dtype=tf.string, shape=1, default_value='NULL') for k in cat_feature})


def build_feature(embedding_size):
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
        f_num = tf.feature_column.numeric_column(i,
                                                 default_value=0,
                                                 normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
        f_bucket = tf.feature_column.bucketized_column(f_num, j)
        f_embedding = tf.feature_column.embedding_column(f_bucket, embedding_size)

        # TODO: With duplicated one-hot or not?
        # linear_feature.append(tf.feature_column.indicator_column(f_bucket))

        linear_feature_columns.append(f_num)
        embedding_feature_columns.append(f_embedding)

    for i, j in zip(cat_feature, buckets_cat):
        f_cat = tf.feature_column.categorical_column_with_hash_bucket(key=i, hash_bucket_size=j)

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
    cross_layers = list(map(int, params["cross_layers"].split(',')))

    linear_net = tf.feature_column.input_layer(features, params['linear_feature_columns'])
    embedding_net = tf.feature_column.input_layer(features, params['embedding_feature_columns'])

    with tf.name_scope('linear_net'):
        linear_y = tf.layers.dense(linear_net, 1, activation=tf.nn.relu)

    with tf.name_scope('cin_net'):
        field_nums = []
        hidden_nn_layers = []
        final_len = 0
        final_result = []
        cin_net = tf.reshape(embedding_net,
                             shape=[-1, len(params['embedding_feature_columns']), params['embedding_size']])
        field_nums.append(len(params['embedding_feature_columns']))
        hidden_nn_layers.append(cin_net)

        split_tensor0 = tf.split(hidden_nn_layers[0], params['embedding_size'] * [1], 2)

        for idx, layer_size in enumerate(cross_layers):
            split_tensor = tf.split(hidden_nn_layers[-1], params['embedding_size'] * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m,
                                      shape=[params['embedding_size'], -1, field_nums[0] * field_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            filters = tf.get_variable(name="f_" + str(idx),
                                      shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                      dtype=tf.float32)

            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')

            # Add bias
            b = tf.get_variable(name="f_b" + str(idx),
                                shape=[layer_size],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            curr_out = tf.nn.bias_add(curr_out, b)

            # Activation
            curr_out = tf.nn.relu(curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            # Connect direct
            direct_connect = curr_out
            next_hidden = curr_out
            final_len += layer_size
            field_nums.append(int(layer_size))

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)
        cin_y = tf.layers.dense(result, 1, activation=tf.nn.relu)

    with tf.name_scope('dnn_net'):

        embedding_net = tf.feature_column.input_layer(features, params['embedding_feature_columns'])
        dnn_net = tf.reshape(embedding_net,
                             shape=[-1, len(params['embedding_feature_columns']) * params['embedding_size']])
        for i in range(len(layers)):
            dnn_net = tf.layers.dense(dnn_net, i, activation=tf.nn.relu)
            dnn_net = tf.layers.batch_normalization(dnn_net, training=(mode == estimator.ModeKeys.TRAIN))
            dnn_net = tf.layers.dropout(dnn_net, rate=params['dropout'], training=(mode == estimator.ModeKeys.TRAIN))
        dnn_y = tf.layers.dense(dnn_net, 1, activation=tf.nn.relu)

    logits = tf.concat([linear_y, cin_y, dnn_y], axis=-1)
    logits = tf.layers.dense(logits, units=1)
    pred = tf.sigmoid(logits)

    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: estimator.export.PredictOutput(
            predictions)}

    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=tf.cast(labels, tf.float32))
    )
    eval_metric_ops = {
        "AUC": tf.metrics.auc(labels, pred),
        'Accuracy': tf.metrics.accuracy(labels, pred)
    }

    if mode == estimator.ModeKeys.EVAL:
        return estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == estimator.ModeKeys.TRAIN:
        return estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def main(_):
    # tf.enable_eager_execution()
    # dataset = input_fn(['part-r-00000'], 1)
    # for raw_record in dataset.take(1):
    #     print(repr(raw_record))

    data_dir = FLAGS.train_path
    data_files = []
    for i in range(FLAGS.train_parts):
        data_files.append(data_dir + 'part-r-{:0>5}'.format(i))

    train_files = data_files[:-FLAGS.eval_parts]
    eval_files = data_files[-FLAGS.eval_parts:]

    print(eval_files)

    test_files = []
    for i in range(FLAGS.test_parts):
        test_files.append(FLAGS.test_path + 'part-r-{:0>5}'.format(i))

    linear_feature_columns, embedding_feature_columns = build_feature(FLAGS.embedding_size)

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
        "deep_layers": FLAGS.deep_layers,
        "cross_layers": FLAGS.cross_layers
    }

    xdeepfm = estimator.Estimator(
        model_fn=model_fn,
        model_dir='./models/xdeepfm',
        params=model_params,
        config=config
    )

    if FLAGS.task_type == 'train':
        train_spec = estimator.TrainSpec(input_fn=lambda: input_fn(
            train_files,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            need_shuffle=True))
        eval_spec = estimator.EvalSpec(input_fn=lambda: input_fn(
            eval_files,
            num_epochs=-1,
            batch_size=FLAGS.batch_size), steps=200, start_delay_secs=1, throttle_secs=5)
        start = time()
        estimator.train_and_evaluate(xdeepfm, train_spec, eval_spec)
        elapsed = (time() - start)
        tf.logging.info("Training time used: {0}ms".format(round(elapsed * 1000, 2)))
    elif FLAGS.task_type == 'eval':
        xdeepfm.evaluate(input_fn=lambda: input_fn(eval_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'predict':
        p = xdeepfm.predict(input_fn=lambda: input_fn(eval_files, num_epochs=1, batch_size=FLAGS.batch_size))
        tf.logging.info('done predit')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
