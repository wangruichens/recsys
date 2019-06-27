import tensorflow as tf
from tensorflow import feature_column
import sys
from tensorflow_estimator import estimator
import os
from time import time

# os.environ["CUDA_VISIBLE_DEVICES"] = ''
# tf.enable_eager_execution()


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout rate")
tf.app.flags.DEFINE_string("task_type", 'train', "Task type {train, infer, eval, export}")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs")
tf.app.flags.DEFINE_string("deep_layers", '200,200,200', "deep layers")
tf.app.flags.DEFINE_string("dataset_path", '/home/wangrc/deepfm_data/', "Data path")
tf.app.flags.DEFINE_integer("dataset_parts", 100, "Tfrecord counts")
tf.app.flags.DEFINE_integer("dataset_eval", 1, "Eval tfrecord")
tf.app.flags.DEFINE_string("export_path", './export/', "Model export path")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 10, "Log_step_count_steps")
tf.app.flags.DEFINE_integer("save_checkpoints_steps", 500, "save_checkpoints_steps")
tf.app.flags.DEFINE_boolean("mirror", True, "Mirrored Strategy")

feature_description = {
    'label': tf.FixedLenFeature((), tf.int64),

    "u_id": tf.FixedLenFeature(dtype=tf.int64, shape=1),
    "i_id": tf.FixedLenFeature(dtype=tf.int64, shape=1),
}
categoryFeatureNa = '####'


def build_model_columns(embedding_size):
    linear_feature_columns = []
    embedding_feature_columns = []

    u_id = feature_column.categorical_column_with_hash_bucket('u_id', 500000, dtype=tf.dtypes.int64)
    u_id_embedded = feature_column.embedding_column(u_id, embedding_size)
    linear_feature_columns.append(feature_column.indicator_column(u_id))
    embedding_feature_columns.append(u_id_embedded)

    i_id = feature_column.categorical_column_with_hash_bucket('i_id', 100000, dtype=tf.dtypes.int64)
    i_id_embedded = feature_column.embedding_column(i_id, embedding_size)
    linear_feature_columns.append(feature_column.indicator_column(i_id))
    embedding_feature_columns.append(i_id_embedded)

    return linear_feature_columns, embedding_feature_columns


def _parse_example(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features=feature_description)
    labels = features.pop('label')
    return features, labels


def input_fn(filenames, batch_size=32, num_epochs=-1, need_shuffle=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_example, num_parallel_calls=4).batch(batch_size)
    if need_shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    # https://www.tensorflow.org/guide/performance/datasets#map_and_cache
    # The Dataset.prefetch(m) transformation prefetches m elements of its direct input.
    # Since its direct input is dataset.batch(n) and each element of that dataset is a batch (of n elements),
    # it will prefetch m batches.
    dataset = dataset.prefetch(buffer_size=100).repeat(num_epochs)
    return dataset


def model_fn(features, labels, mode, params):
    """
    Build model for estimator.
    @param features:
    @param labels:
    @param mode:
    @param params:
    @return:
    """
    layers = list(map(int, params["deep_layers"].split(',')))

    embedding_feature_columns = params['embedding_feature_columns']
    embedding_features = feature_column.input_layer(features, embedding_feature_columns)

    linear_feature_columns = params['linear_feature_columns']
    linear_features = feature_column.input_layer(features, linear_feature_columns)

    with tf.variable_scope('first-order'):
        y_1d = tf.layers.dense(linear_features, 1, activation=tf.nn.relu)
    with tf.variable_scope('second-order'):
        fm_net = tf.reshape(embedding_features,
                            [-1, len(params['embedding_feature_columns']), params['embedding_size']])

        fm_net_sum_square = tf.square(tf.reduce_sum(fm_net, axis=1))
        fm_net_square_sum = tf.reduce_sum(tf.square(fm_net), axis=1)
        y_2d = 0.5 * tf.reduce_sum(tf.subtract(fm_net_sum_square, fm_net_square_sum), axis=1, keep_dims=True)

    with tf.variable_scope('dnn'):
        dnn_net = tf.reshape(embedding_features,
                             shape=[-1, len(params['embedding_feature_columns']) * params['embedding_size']])
        for i in layers:
            dnn_net = tf.layers.dense(dnn_net, i, activation=tf.nn.relu)

            dnn_net = tf.layers.batch_normalization(dnn_net, training=(mode == estimator.ModeKeys.TRAIN))
            dnn_net = tf.layers.dropout(dnn_net, rate=params['dropout'], training=(mode == estimator.ModeKeys.TRAIN))
        y_dnn = tf.layers.dense(dnn_net, 1, activation=tf.nn.relu)

    logits = tf.concat([y_1d, y_2d, y_dnn], axis=-1)
    logits = tf.layers.dense(logits, units=1)
    logits = tf.reshape(logits, (-1,))
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
        'Accuracy': tf.metrics.accuracy(labels, predictions=tf.round(pred))
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
    linear_feature_columns, embedding_feature_columns = build_model_columns(FLAGS.embedding_size)

    # session_config = tf.ConfigProto(log_device_placement=True)
    # session_config.gpu_options.allow_growth = True

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

    deepfm = estimator.Estimator(
        model_fn=model_fn,
        model_dir='./models/deepfm',
        params=model_params,
        config=config
    )

    data_dir = FLAGS.dataset_path
    data_files = []
    for i in range(FLAGS.dataset_parts):
        data_files.append(data_dir + 'part-r-{:0>5}'.format(i))

    train_files = data_files[:-FLAGS.dataset_eval]
    eval_files = data_files[-FLAGS.dataset_eval:]
    # hook = estimator.ProfilerHook(save_steps=10, output_dir='./time/', show_memory=True, show_dataflow=True)

    if FLAGS.task_type == 'train':
        train_spec = estimator.TrainSpec(input_fn=lambda: input_fn(
            train_files,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            need_shuffle=True))
        eval_spec = estimator.EvalSpec(input_fn=lambda: input_fn(
            eval_files,
            num_epochs=1,
            batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1, throttle_secs=5)
        start = time()
        estimator.train_and_evaluate(deepfm, train_spec, eval_spec)
        elapsed = (time() - start)
        print("Training time used: {0}ms".format(round(elapsed * 1000, 2)))
    elif FLAGS.task_type == 'eval':
        deepfm.evaluate(input_fn=lambda: input_fn(eval_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'predict':
        p = deepfm.predict(input_fn=lambda: input_fn(eval_files, num_epochs=1, batch_size=FLAGS.batch_size))
        print('done predit')
        # tf.data.Dataset.from_tensor_slices()
        # for i in p:
        #     print(i["prob"])

    feature_description.pop('label')
    serving_fn = estimator.export.build_parsing_serving_input_receiver_fn(feature_description)

    # features = {
    #     "u_id": tf.placeholder(dtype=tf.int32, shape=1, name='u_id'),
    #     "i_id": tf.placeholder(dtype=tf.int32, shape=1, name='i_id'),
    # }
    # serving_fn = estimator.export.build_raw_serving_input_receiver_fn(features)

    deepfm.export_savedmodel(
        export_dir_base=FLAGS.export_path,
        serving_input_receiver_fn=serving_fn,
        as_text=True,
    )
    tf.logging.info('Model exported.')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
