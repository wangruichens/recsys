import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.apis import prediction_log_pb2
import grpc
from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=deepfm \
#   --model_base_path="/home/wangrc/Desktop/"

tf.app.flags.DEFINE_string('server', '172.17.18.9:26389',
                           'Server host:port.')
tf.app.flags.DEFINE_string('test_file', 'test.csv',
                           'Test file.')
tf.app.flags.DEFINE_string('model', 'deepfm',
                           'Model name.')
tf.app.flags.DEFINE_string('signature_name', 'serving_default',
                           'Signature name.')
tf.app.flags.DEFINE_integer('batch_size', 200,
                            'Batch size.')
FLAGS = tf.app.flags.FLAGS


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gen_data(filename, size=1000):
    df = pd.read_csv(filename)
    df = df.sample(n=size)
    df_y = df.label
    df = df.drop(['label'], axis=1)

    batching = []
    for _, row in df.iterrows():
        feature_dict = {"u_id": _int_feature(row['u_id']),
                        "i_id": _int_feature(row['i_id'])}
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example.SerializeToString()
        batching.append(serialized)
    return batching, df_y.values


def main(_):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Warm up
    request_w = predict_pb2.PredictRequest()
    test_batch, y_true = gen_data(FLAGS.test_file, 1)
    request_w.model_spec.name = FLAGS.model
    request_w.model_spec.signature_name = FLAGS.signature_name
    request_w.inputs['examples'].CopyFrom(tf.make_tensor_proto(test_batch, shape=[len(test_batch)]))
    prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request_w))

    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = FLAGS.signature_name

    test_batch, y_true = gen_data(FLAGS.test_file, FLAGS.batch_size)
    request.inputs['examples'].CopyFrom(
        tf.make_tensor_proto(test_batch, shape=[len(test_batch)]))

    start = time()
    result_future = stub.Predict.future(request, 10.0)
    elapsed = (time() - start)
    prediction = result_future.result().outputs['prob']
    # print(prediction)
    print('Batch size: ', FLAGS.batch_size)
    print('Predict AUC: ', roc_auc_score(y_true, prediction.float_val))
    print("Predict time used: {0}ms".format(round(elapsed * 1000, 2)))


if __name__ == '__main__':
    tf.app.run()
