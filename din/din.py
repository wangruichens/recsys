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
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout rate")
tf.app.flags.DEFINE_string("task_type", 'train', "Task type {train, infer, eval, export}")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs")
tf.app.flags.DEFINE_string("deep_layers", '100,100', "deep layers")
tf.app.flags.DEFINE_string("din_layers", '80,40', "cross layers")

tf.app.flags.DEFINE_integer("pkg_count", 2869, "Number of pkg")
tf.app.flags.DEFINE_integer("pkgc_count", 15, "Number of pkgc")
tf.app.flags.DEFINE_integer("ssid_count", 27, "Number of ssid")
tf.app.flags.DEFINE_integer("oper_count", 2052, "Number of oper")

tf.app.flags.DEFINE_string("train_path", '/home/wangrc/din_dataset/', "Data path")
tf.app.flags.DEFINE_integer("train_parts", 11, "Tfrecord counts")
tf.app.flags.DEFINE_integer("eval_parts", 2, "Eval tfrecord")

tf.app.flags.DEFINE_string("test_path", '/home/wangrc/criteo_data/test/', "Test path")
tf.app.flags.DEFINE_integer("test_parts", 15, "Tfrecord counts")

tf.app.flags.DEFINE_string("export_path", './export/', "Model export path")
tf.app.flags.DEFINE_string("model_path", 'hdfs://cluster/user/wangrc/zs/din_model/', "Model export path")
tf.app.flags.DEFINE_integer("batch_size", 256, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 100, "Log_step_count_steps")
tf.app.flags.DEFINE_integer("eval_steps", 200, "Eval_steps")

tf.app.flags.DEFINE_integer("save_checkpoints_steps", 1000, "save_checkpoints_steps")
tf.app.flags.DEFINE_boolean("mirror", True, "Mirrored Strategy")

# 'imei', 'pkg_hist', 'pkgc_hist', 'ssid_hist', 'soper_hist', 'pkg', 'pkgc', 'ssid', 'soper', 'label'
feature_description = {
    "label": tf.FixedLenFeature([], dtype=tf.int64),
    # "u_id": tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
    "i_id": tf.FixedLenFeature([], dtype=tf.int64),
    "i_cate": tf.FixedLenFeature([], dtype=tf.int64),
    "u_iid_seq": tf.VarLenFeature(tf.int64),
    "u_icat_seq": tf.VarLenFeature(tf.int64),
}

def _parse_examples(serial_exmp):
	features = tf.parse_single_example(serial_exmp, features = feature_description)

	labels = features.pop('label')
	features['u_iid_seq'] = tf.sparse_tensor_to_dense(features['u_iid_seq'])
	features['u_icat_seq'] = tf.sparse_tensor_to_dense(features['u_icat_seq'])
	# features['ssid_hist'] = tf.sparse_tensor_to_dense(features['ssid_hist'])
	# features['oper_hist'] = tf.sparse_tensor_to_dense(features['oper_hist'])
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
	dataset = dataset.map(_parse_examples, num_parallel_calls = 6).batch(batch_size)
	if need_shuffle:
		dataset = dataset.shuffle(buffer_size = 1000)
	dataset = dataset.prefetch(buffer_size = 1000).repeat(num_epochs)
	return dataset


def model_fn(features, labels, mode, params):
	# imei_emb_w = tf.get_variable("user_emb_w", [params['imei_count'], params['hidden_units']])
	attention_layers = [80, 40]
	mlp_layers = [100, 50, 20]

	pkg_w = tf.get_variable("i_item", [63002],initializer=tf.constant_initializer(0.0))
	pkg_emb_w = tf.get_variable("i_id", [63002, params['embedding_size']],initializer = tf.glorot_normal_initializer())
	pkgc_emb_w = tf.get_variable("i_cate", [802, params['embedding_size']],initializer = tf.glorot_normal_initializer())

	# pkg_emb_w = tf.get_variable("pkg_emb_w", [2360, params['embedding_size']])
	# pkgc_emb_w = tf.get_variable("pkgc_emb_w", [100, params['embedding_size']])
	# ssid_emb_w = tf.get_variable("ssid_emb_w", [100, params['embedding_size']])
	# soper_emb_w = tf.get_variable("soper_emb_w", [100, params['embedding_size']])
	i_b =tf.gather(pkg_w, features['i_id'])
	with tf.variable_scope("embedding_layer"):
		# imei_emb = tf.nn.embedding_lookup(imei_emb_w, features['imei_index'])

		pkg_emb = tf.nn.embedding_lookup(pkg_emb_w, features['i_id'])
		pkgc_emb = tf.nn.embedding_lookup(pkgc_emb_w, features['i_cate'])

		def _attention(feat_emb_w, hist_ids, item_emb):
			dense_ids = hist_ids
			dense_emb = tf.nn.embedding_lookup(feat_emb_w, dense_ids)  # None * P * K

			dense_mask = tf.expand_dims(tf.cast(dense_ids > 0, tf.float32),axis = -1)
			# dense_mask = tf.sequence_mask(dense_ids, ?)  # None * P
			padded_dim = tf.shape(dense_ids)[1]  # P
			hist_emb = tf.reshape(dense_emb, shape = [-1, params['embedding_size']])
			query_emb = tf.reshape(tf.tile(item_emb, [1, padded_dim]), shape = [-1, params['embedding_size']])
			# None * K --> (None * P) * K     注意跟dense_emb reshape顺序保持一致

			att_net = tf.concat([hist_emb, query_emb, hist_emb * query_emb, hist_emb - query_emb],axis = 1)  # (None * P) * 3K
			for i in attention_layers:
				att_net = tf.layers.dense(att_net, units = i, activation = tf.nn.relu)
				# att_net = tf.layers.batch_normalization(att_net, training = (mode == estimator.ModeKeys.TRAIN))
				att_net = tf.layers.dropout(att_net, rate = params['dropout'], training = (mode == estimator.ModeKeys.TRAIN))

			att_wgt = tf.layers.dense(att_net, units = 1, activation = None)
			att_wgt = tf.reshape(att_wgt, shape = [-1, padded_dim, 1])  # None * P * 1
			wgt_emb = tf.multiply(dense_emb, att_wgt)  # None * P * K
			# dense_mask
			wgt_emb = tf.reduce_sum(tf.multiply(wgt_emb, dense_mask), 1)  # None * K
			return wgt_emb

		pkg_emb_h = _attention(pkg_emb_w, features['u_iid_seq'], pkg_emb)
		pkgc_emb_h = _attention(pkgc_emb_w, features['u_icat_seq'], pkgc_emb)

	with tf.variable_scope("mlp_layer"):
		net = tf.concat([pkg_emb, pkg_emb_h, pkgc_emb_h], axis = 1)

		for units in mlp_layers:
			net = tf.layers.dense(net, units = units, activation = tf.nn.relu)
			# net = tf.layers.batch_normalization(net, training = (mode == estimator.ModeKeys.TRAIN))
			net = tf.layers.dropout(net, rate = params['dropout'], training = (mode == estimator.ModeKeys.TRAIN))

		logits = tf.layers.dense(net, units = 1, activation = None)
		logits = tf.reshape(logits, (-1,))  +i_b
		pred = tf.sigmoid(logits)

		predictions = {"prob": pred}
		export_outputs = {
			tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: estimator.export.PredictOutput(
				predictions)}

		if mode == estimator.ModeKeys.PREDICT:
			return estimator.EstimatorSpec(
				mode = mode,
				predictions = predictions,
				export_outputs = export_outputs)

		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits = logits,
			labels = tf.cast(labels, tf.float32))
		)
		# loss += 0.01 * tf.nn.l2_loss(pkg_emb_w)
		# loss += 0.01 * tf.nn.l2_loss(pkgc_emb_w)

		eval_metric_ops = {
			"AUC": tf.metrics.auc(labels, pred),
			'Accuracy': tf.metrics.accuracy(labels, predictions = tf.round(pred))
		}

		if mode == estimator.ModeKeys.EVAL:
			return estimator.EstimatorSpec(
				mode = mode,
				predictions = predictions,
				loss = loss,
				eval_metric_ops = eval_metric_ops)

		optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
		train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())

		if mode == estimator.ModeKeys.TRAIN:
			return estimator.EstimatorSpec(
				mode = mode,
				predictions = predictions,
				loss = loss,
				train_op = train_op)


def main(_):
	# tf.enable_eager_execution()
	# dataset = input_fn(['train2'], 1)
	# for raw_record in dataset.take(1):
	# 	print(raw_record[1])

	data_dir = FLAGS.train_path
	data_files = []
	for i in range(FLAGS.train_parts):
		data_files.append(data_dir + 'part-r-{:0>5}'.format(i))

	train_files = data_files[:-FLAGS.eval_parts]
	eval_files = data_files[-FLAGS.eval_parts:]

	train_files = ['train2']
	eval_files =['valid2']

	test_files = []
	for i in range(FLAGS.test_parts):
		test_files.append(FLAGS.test_path + 'part-r-{:0>5}'.format(i))

	distribute_strategy = None
	if FLAGS.mirror:
		distribute_strategy = tf.distribute.MirroredStrategy()

	config = estimator.RunConfig(
		save_checkpoints_steps = FLAGS.save_checkpoints_steps,
		keep_checkpoint_max = 5,
		log_step_count_steps = FLAGS.log_steps,
		save_summary_steps = 200,
		train_distribute = distribute_strategy,
		eval_distribute = distribute_strategy
	)

	model_params = {
		'pkg_count': FLAGS.pkg_count,
		'pkgc_count': FLAGS.pkgc_count,
		'ssid_count': FLAGS.ssid_count,
		'oper_count': FLAGS.oper_count,
		'embedding_size': FLAGS.embedding_size,
		"learning_rate": FLAGS.learning_rate,
		"dropout": FLAGS.dropout,
		"deep_layers": FLAGS.deep_layers,
	}

	din = estimator.Estimator(
		model_fn = model_fn,
		model_dir = './models/',
		params = model_params,
		config = config
	)

	if FLAGS.task_type == 'train':
		train_spec = estimator.TrainSpec(input_fn = lambda: input_fn(
			train_files,
			num_epochs = FLAGS.num_epochs,
			batch_size = FLAGS.batch_size,
			need_shuffle = True))
		eval_spec = estimator.EvalSpec(input_fn = lambda: input_fn(
			eval_files,
			num_epochs = -1,
			batch_size = FLAGS.batch_size), steps = FLAGS.eval_steps, start_delay_secs = 1, throttle_secs = 5)
		start = time()
		estimator.train_and_evaluate(din, train_spec, eval_spec)
		elapsed = (time() - start)
		tf.logging.info("Training time used: {0}ms".format(round(elapsed * 1000, 2)))
	elif FLAGS.task_type == 'eval':
		din.evaluate(input_fn = lambda: input_fn(eval_files, num_epochs = 1, batch_size = FLAGS.batch_size),
		             steps = FLAGS.eval_steps * 10)
	elif FLAGS.task_type == 'predict':
		p = din.predict(input_fn = lambda: input_fn(eval_files, num_epochs = 1, batch_size = FLAGS.batch_size))
		tf.logging.info('done predit')


# feature_description.pop('label')
# serving_fn = estimator.export.build_parsing_serving_input_receiver_fn(feature_description)
#
# # features = {
# #     "u_id": tf.placeholder(dtype=tf.int32, shape=1, name='u_id'),
# #     "i_id": tf.placeholder(dtype=tf.int32, shape=1, name='i_id'),
# # }
# # serving_fn = estimator.export.build_raw_serving_input_receiver_fn(features)
#
# din.export_savedmodel(
# 	export_dir_base = FLAGS.export_path,
# 	serving_input_receiver_fn = serving_fn,
# 	as_text = True,
# )
# tf.logging.info('Model exported.')


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
