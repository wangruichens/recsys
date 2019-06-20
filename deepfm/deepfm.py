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
    "i_channel": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_brand": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_operator": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_activelevel": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_age": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_marriage": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_sex": tf.FixedLenFeature(dtype=tf.string, shape=1, ),
    "u_sex_age": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_sex_marriage": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_age_marriage": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "i_hot_news": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "i_is_recommend": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "i_info_exposed_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_info_clicked_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_info_ctr": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_cate_exposed_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_cate_clicked_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_category_ctr": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_7": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_7": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_7": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_14": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_14": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_14": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_user_flavor": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at2": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at4": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at5": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_mini_img_size": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_comment_count": tf.FixedLenFeature(dtype=tf.float32, shape=1),
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
    i_channel = feature_column.categorical_column_with_vocabulary_list(
        'i_channel',
        vocabulary_list=[
            categoryFeatureNa,
            "bendizhengwu", "caijing", "chongwu", "dongman", "fangchan",
            "guoji", "guonei", "jiaju", "jiankang", "jiaoyu",
            "junshi", "keji", "lishi", "lvyou", "nanchang",
            "qiche", "qinggan", "sannong", "shehui", "shishang",
            "tiyu", "vcaijing", "vgaoxiao", "vpaike", "vshishang",
            "vtiyu", "vyule", "vzixun", "weikandian", "xiaohua",
            "xingzuo", "xinwen", "youxi", "yuer", "yule",
            "ziran"
        ],
        dtype=tf.string,
        default_value=0,
    )
    linear_feature_columns.append(feature_column.indicator_column(i_channel))
    i_channel = feature_column.embedding_column(i_channel, embedding_size)
    embedding_feature_columns.append(i_channel)

    i_info_exposed_amt = feature_column.numeric_column(
        'i_info_exposed_amt', default_value=0.0)
    i_info_exposed_amt = feature_column.bucketized_column(i_info_exposed_amt,
                                                          [1e-60, 1.0, 11.0, 36.0, 75.0, 132.0, 216.0, 340.0, 516.0,
                                                           746.0, 1064.0, 1680.0, 2487.0, 3656.0, 5138.0, 7837.0,
                                                           11722.0, 18993.0, 117513.0])
    linear_feature_columns.append(
        feature_column.indicator_column(i_info_exposed_amt))
    i_info_exposed_amt = feature_column.embedding_column(
        i_info_exposed_amt, embedding_size)
    embedding_feature_columns.append(i_info_exposed_amt)

    i_info_exposed_amt = feature_column.numeric_column('i_info_exposed_amt', default_value=0.0,
                                                       normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_info_exposed_amt)

    i_info_clicked_amt = feature_column.numeric_column(
        'i_info_clicked_amt', default_value=0.0)
    i_info_clicked_amt = feature_column.bucketized_column(i_info_clicked_amt,
                                                          [1e-60, 1.0, 5.0, 12.0, 23.0, 37.0, 57.0, 96.0, 139.0, 209.0,
                                                           299.0, 442.0, 647.0, 929.0, 1386.0, 1993.0, 3312.0, 17578.0])
    linear_feature_columns.append(
        feature_column.indicator_column(i_info_clicked_amt))
    i_info_clicked_amt = feature_column.embedding_column(
        i_info_clicked_amt, embedding_size)
    embedding_feature_columns.append(i_info_clicked_amt)

    i_info_clicked_amt = feature_column.numeric_column('i_info_clicked_amt', default_value=0.0,
                                                       normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_info_clicked_amt)

    i_info_ctr = feature_column.numeric_column('i_info_ctr', default_value=0.0)
    i_info_ctr = feature_column.bucketized_column(i_info_ctr,
                                                  [1e-60, 0.05272500000000001, 0.098039, 0.122831, 0.139318, 0.154176,
                                                   0.166484, 0.175818, 0.184211, 0.19266100000000003, 0.20089, 0.2098,
                                                   0.218659, 0.227273, 0.236842, 0.25, 0.280294, 1.0])
    linear_feature_columns.append(feature_column.indicator_column(i_info_ctr))
    i_info_ctr = feature_column.embedding_column(i_info_ctr, embedding_size)
    embedding_feature_columns.append(i_info_ctr)

    i_info_ctr = feature_column.numeric_column('i_info_ctr', default_value=0.0,
                                               normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_info_ctr)

    i_cate_exposed_amt = feature_column.numeric_column(
        'i_cate_exposed_amt', default_value=0.0)
    i_cate_exposed_amt = feature_column.bucketized_column(i_cate_exposed_amt,
                                                          [1e-60, 2620264.0, 2893466.0, 4885968.0, 5074062.0, 5648085.0,
                                                           9389498.0, 9900840.0, 10462611.0, 14308882.0, 17151668.0,
                                                           33770813.0])
    linear_feature_columns.append(
        feature_column.indicator_column(i_cate_exposed_amt))
    i_cate_exposed_amt = feature_column.embedding_column(
        i_cate_exposed_amt, embedding_size)
    embedding_feature_columns.append(i_cate_exposed_amt)

    i_cate_exposed_amt = feature_column.numeric_column('i_cate_exposed_amt', default_value=0.0,
                                                       normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_cate_exposed_amt)

    i_cate_clicked_amt = feature_column.numeric_column(
        'i_cate_clicked_amt', default_value=0.0)
    i_cate_clicked_amt = feature_column.bucketized_column(i_cate_clicked_amt,
                                                          [1e-60, 315824.0, 454217.0, 739206.0, 994371.0, 1072646.0,
                                                           1255534.0, 1546660.0, 1613939.0, 2640503.0, 2841787.0,
                                                           6283496.0])
    linear_feature_columns.append(
        feature_column.indicator_column(i_cate_clicked_amt))
    i_cate_clicked_amt = feature_column.embedding_column(
        i_cate_clicked_amt, embedding_size)
    embedding_feature_columns.append(i_cate_clicked_amt)

    i_cate_clicked_amt = feature_column.numeric_column('i_cate_clicked_amt', default_value=0.0,
                                                       normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_cate_clicked_amt)

    i_category_ctr = feature_column.numeric_column(
        'i_category_ctr', default_value=0.0)
    i_category_ctr = feature_column.bucketized_column(i_category_ctr,
                                                      [1e-60, 0.11518699999999997, 0.133717, 0.151292, 0.154258,
                                                       0.15621500000000002, 0.162051, 0.165686, 0.184536, 0.186063,
                                                       0.189913, 0.195971])
    linear_feature_columns.append(
        feature_column.indicator_column(i_category_ctr))
    i_category_ctr = feature_column.embedding_column(
        i_category_ctr, embedding_size)
    embedding_feature_columns.append(i_category_ctr)

    i_category_ctr = feature_column.numeric_column('i_category_ctr', default_value=0.0,
                                                   normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_category_ctr)

    c_uid_type_ctr_1 = feature_column.numeric_column(
        'c_uid_type_ctr_1', default_value=0.0)
    c_uid_type_ctr_1 = feature_column.bucketized_column(c_uid_type_ctr_1,
                                                        [1e-60, 0.06666699999999999, 0.14285699999999998,
                                                         0.22222199999999998, 0.333333, 0.45, 0.6363640000000002, 1.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_ctr_1))
    c_uid_type_ctr_1 = feature_column.embedding_column(
        c_uid_type_ctr_1, embedding_size)
    embedding_feature_columns.append(c_uid_type_ctr_1)

    c_uid_type_ctr_1 = feature_column.numeric_column('c_uid_type_ctr_1', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_ctr_1)

    c_uid_type_clicked_amt_1 = feature_column.numeric_column(
        'c_uid_type_clicked_amt_1', default_value=0.0)
    c_uid_type_clicked_amt_1 = feature_column.bucketized_column(c_uid_type_clicked_amt_1,
                                                                [1e-60, 1.0, 2.0, 3.0, 6.0, 11.0, 351.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_clicked_amt_1))
    c_uid_type_clicked_amt_1 = feature_column.embedding_column(
        c_uid_type_clicked_amt_1, embedding_size)
    embedding_feature_columns.append(c_uid_type_clicked_amt_1)

    c_uid_type_clicked_amt_1 = feature_column.numeric_column('c_uid_type_clicked_amt_1', default_value=0.0,
                                                             normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_clicked_amt_1)

    c_uid_type_exposed_amt_1 = feature_column.numeric_column(
        'c_uid_type_exposed_amt_1', default_value=0.0)
    c_uid_type_exposed_amt_1 = feature_column.bucketized_column(c_uid_type_exposed_amt_1,
                                                                [1e-60, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 12.0, 16.0, 23.0,
                                                                 41.0, 988.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_exposed_amt_1))
    c_uid_type_exposed_amt_1 = feature_column.embedding_column(
        c_uid_type_exposed_amt_1, embedding_size)
    embedding_feature_columns.append(c_uid_type_exposed_amt_1)

    c_uid_type_exposed_amt_1 = feature_column.numeric_column('c_uid_type_exposed_amt_1', default_value=0.0,
                                                             normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_exposed_amt_1)

    c_uid_type_ctr_3 = feature_column.numeric_column(
        'c_uid_type_ctr_3', default_value=0.0)
    c_uid_type_ctr_3 = feature_column.bucketized_column(c_uid_type_ctr_3, [1e-60, 0.012820999999999999, 0.058824, 0.1,
                                                                           0.14285699999999998, 0.2, 0.253968, 0.333333,
                                                                           0.4426229999999999, 0.6, 1.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_ctr_3))
    c_uid_type_ctr_3 = feature_column.embedding_column(
        c_uid_type_ctr_3, embedding_size)
    embedding_feature_columns.append(c_uid_type_ctr_3)

    c_uid_type_ctr_3 = feature_column.numeric_column('c_uid_type_ctr_3', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_ctr_3)

    c_uid_type_clicked_amt_3 = feature_column.numeric_column(
        'c_uid_type_clicked_amt_3', default_value=0.0)
    c_uid_type_clicked_amt_3 = feature_column.bucketized_column(c_uid_type_clicked_amt_3,
                                                                [1e-60, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 17.0, 30.0,
                                                                 1084.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_clicked_amt_3))
    c_uid_type_clicked_amt_3 = feature_column.embedding_column(
        c_uid_type_clicked_amt_3, embedding_size)
    embedding_feature_columns.append(c_uid_type_clicked_amt_3)

    c_uid_type_clicked_amt_3 = feature_column.numeric_column('c_uid_type_clicked_amt_3', default_value=0.0,
                                                             normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_clicked_amt_3)

    c_uid_type_exposed_amt_3 = feature_column.numeric_column(
        'c_uid_type_exposed_amt_3', default_value=0.0)
    c_uid_type_exposed_amt_3 = feature_column.bucketized_column(c_uid_type_exposed_amt_3,
                                                                [1e-60, 1.0, 2.0, 4.0, 6.0, 9.0, 13.0, 17.0, 22.0, 28.0,
                                                                 36.0, 48.0, 68.0, 116.0, 2428.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_exposed_amt_3))
    c_uid_type_exposed_amt_3 = feature_column.embedding_column(
        c_uid_type_exposed_amt_3, embedding_size)
    embedding_feature_columns.append(c_uid_type_exposed_amt_3)

    c_uid_type_exposed_amt_3 = feature_column.numeric_column('c_uid_type_exposed_amt_3', default_value=0.0,
                                                             normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_exposed_amt_3)

    c_uid_type_ctr_7 = feature_column.numeric_column(
        'c_uid_type_ctr_7', default_value=0.0)
    c_uid_type_ctr_7 = feature_column.bucketized_column(c_uid_type_ctr_7,
                                                        [1e-60, 0.033898000000000005, 0.061224, 0.090909, 0.124444,
                                                         0.162162, 0.208333, 0.266667, 0.333333, 0.433333, 0.575342,
                                                         1.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_ctr_7))
    c_uid_type_ctr_7 = feature_column.embedding_column(
        c_uid_type_ctr_7, embedding_size)
    embedding_feature_columns.append(c_uid_type_ctr_7)

    c_uid_type_ctr_7 = feature_column.numeric_column('c_uid_type_ctr_7', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_ctr_7)

    c_uid_type_clicked_amt_7 = feature_column.numeric_column(
        'c_uid_type_clicked_amt_7', default_value=0.0)
    c_uid_type_clicked_amt_7 = feature_column.bucketized_column(c_uid_type_clicked_amt_7,
                                                                [1e-60, 1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 16.0, 24.0, 37.0,
                                                                 67.0, 1613.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_clicked_amt_7))
    c_uid_type_clicked_amt_7 = feature_column.embedding_column(
        c_uid_type_clicked_amt_7, embedding_size)
    embedding_feature_columns.append(c_uid_type_clicked_amt_7)

    c_uid_type_clicked_amt_7 = feature_column.numeric_column('c_uid_type_clicked_amt_7', default_value=0.0,
                                                             normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_clicked_amt_7)

    c_uid_type_exposed_amt_7 = feature_column.numeric_column(
        'c_uid_type_exposed_amt_7', default_value=0.0)
    c_uid_type_exposed_amt_7 = feature_column.bucketized_column(c_uid_type_exposed_amt_7,
                                                                [1e-60, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 30.0, 40.0,
                                                                 51.0, 65.0, 84.0, 110.0, 153.0, 256.0, 6136.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_exposed_amt_7))
    c_uid_type_exposed_amt_7 = feature_column.embedding_column(
        c_uid_type_exposed_amt_7, embedding_size)
    embedding_feature_columns.append(c_uid_type_exposed_amt_7)

    c_uid_type_exposed_amt_7 = feature_column.numeric_column('c_uid_type_exposed_amt_7', default_value=0.0,
                                                             normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_exposed_amt_7)

    c_uid_type_ctr_14 = feature_column.numeric_column(
        'c_uid_type_ctr_14', default_value=0.0)
    c_uid_type_ctr_14 = feature_column.bucketized_column(c_uid_type_ctr_14,
                                                         [1e-60, 0.028777, 0.051282000000000015, 0.074205, 0.1,
                                                          0.130841, 0.166667, 0.212121, 0.266667, 0.333333, 0.425,
                                                          0.563265, 1.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_ctr_14))
    c_uid_type_ctr_14 = feature_column.embedding_column(
        c_uid_type_ctr_14, embedding_size)
    embedding_feature_columns.append(c_uid_type_ctr_14)

    c_uid_type_ctr_14 = feature_column.numeric_column('c_uid_type_ctr_14', default_value=0.0,
                                                      normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_ctr_14)

    c_uid_type_clicked_amt_14 = feature_column.numeric_column(
        'c_uid_type_clicked_amt_14', default_value=0.0)
    c_uid_type_clicked_amt_14 = feature_column.bucketized_column(c_uid_type_clicked_amt_14,
                                                                 [1e-60, 1.0, 2.0, 4.0, 6.0, 9.0, 14.0, 20.0, 31.0,
                                                                  47.0, 74.0, 134.0, 3407.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_clicked_amt_14))
    c_uid_type_clicked_amt_14 = feature_column.embedding_column(
        c_uid_type_clicked_amt_14, embedding_size)
    embedding_feature_columns.append(c_uid_type_clicked_amt_14)

    c_uid_type_clicked_amt_14 = feature_column.numeric_column('c_uid_type_clicked_amt_14', default_value=0.0,
                                                              normalizer_fn=lambda x: tf.log(
                                                                  x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_clicked_amt_14)

    c_uid_type_exposed_amt_14 = feature_column.numeric_column(
        'c_uid_type_exposed_amt_14', default_value=0.0)
    c_uid_type_exposed_amt_14 = feature_column.bucketized_column(c_uid_type_exposed_amt_14,
                                                                 [1e-60, 1.0, 3.0, 7.0, 13.0, 20.0, 29.0, 42.0, 58.0,
                                                                  77.0, 100.0, 129.0, 167.0, 221.0, 308.0, 507.0,
                                                                  14466.0])
    linear_feature_columns.append(
        feature_column.indicator_column(c_uid_type_exposed_amt_14))
    c_uid_type_exposed_amt_14 = feature_column.embedding_column(
        c_uid_type_exposed_amt_14, embedding_size)
    embedding_feature_columns.append(c_uid_type_exposed_amt_14)

    c_uid_type_exposed_amt_14 = feature_column.numeric_column('c_uid_type_exposed_amt_14', default_value=0.0,
                                                              normalizer_fn=lambda x: tf.log(
                                                                  x + sys.float_info.epsilon))
    linear_feature_columns.append(c_uid_type_exposed_amt_14)

    c_user_flavor = feature_column.numeric_column(
        'c_user_flavor', default_value=0.0)
    c_user_flavor = feature_column.bucketized_column(c_user_flavor,
                                                     [1e-60, 0.0012, 0.0493, 0.0881, 0.1233, 0.1573, 0.1933, 0.2323,
                                                      0.2777, 0.3366, 0.4182, 0.5487, 0.8213, 4.7547])
    linear_feature_columns.append(
        feature_column.indicator_column(c_user_flavor))
    c_user_flavor = feature_column.embedding_column(c_user_flavor, embedding_size)
    embedding_feature_columns.append(c_user_flavor)

    c_user_flavor = feature_column.numeric_column('c_user_flavor', default_value=0.0,
                                                  normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(c_user_flavor)

    u_activetime_at1 = feature_column.numeric_column(
        'u_activetime_at1', default_value=0.0)
    u_activetime_at1 = feature_column.bucketized_column(
        u_activetime_at1, [1e-60, 1.0, 2.0, 3.0, 6.0, 10.0, 8764.0])
    linear_feature_columns.append(
        feature_column.indicator_column(u_activetime_at1))
    u_activetime_at1 = feature_column.embedding_column(
        u_activetime_at1, embedding_size)
    embedding_feature_columns.append(u_activetime_at1)

    u_activetime_at1 = feature_column.numeric_column('u_activetime_at1', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(u_activetime_at1)

    u_activetime_at2 = feature_column.numeric_column(
        'u_activetime_at2', default_value=0.0)
    u_activetime_at2 = feature_column.bucketized_column(
        u_activetime_at2, [1e-60, 1.0, 3.0, 5.0, 8.0, 7849.0])
    linear_feature_columns.append(
        feature_column.indicator_column(u_activetime_at2))
    u_activetime_at2 = feature_column.embedding_column(
        u_activetime_at2, embedding_size)
    embedding_feature_columns.append(u_activetime_at2)

    u_activetime_at2 = feature_column.numeric_column('u_activetime_at2', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(u_activetime_at2)

    u_activetime_at3 = feature_column.numeric_column(
        'u_activetime_at3', default_value=0.0)
    u_activetime_at3 = feature_column.bucketized_column(
        u_activetime_at3, [1e-60, 1.0, 2.0, 3.0, 5.0, 4075.0])
    linear_feature_columns.append(
        feature_column.indicator_column(u_activetime_at3))
    u_activetime_at3 = feature_column.embedding_column(
        u_activetime_at3, embedding_size)
    embedding_feature_columns.append(u_activetime_at3)

    u_activetime_at3 = feature_column.numeric_column('u_activetime_at3', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(u_activetime_at3)

    u_activetime_at4 = feature_column.numeric_column(
        'u_activetime_at4', default_value=0.0)
    u_activetime_at4 = feature_column.bucketized_column(
        u_activetime_at4, [1e-60, 1.0, 2.0, 3.0, 5.0, 9.0, 10641.0])
    linear_feature_columns.append(
        feature_column.indicator_column(u_activetime_at4))
    u_activetime_at4 = feature_column.embedding_column(
        u_activetime_at4, embedding_size)
    embedding_feature_columns.append(u_activetime_at4)

    u_activetime_at4 = feature_column.numeric_column('u_activetime_at4', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(u_activetime_at4)

    u_activetime_at5 = feature_column.numeric_column(
        'u_activetime_at5', default_value=0.0)
    u_activetime_at5 = feature_column.bucketized_column(u_activetime_at5,
                                                        [1e-60, 1.0, 3.0, 6.0, 10.0, 17.0, 22105.0])
    linear_feature_columns.append(
        feature_column.indicator_column(u_activetime_at5))
    u_activetime_at5 = feature_column.embedding_column(
        u_activetime_at5, embedding_size)
    embedding_feature_columns.append(u_activetime_at5)

    u_activetime_at5 = feature_column.numeric_column('u_activetime_at5', default_value=0.0,
                                                     normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(u_activetime_at5)

    i_mini_img_size = feature_column.numeric_column(
        'i_mini_img_size', default_value=0.0)
    i_mini_img_size = feature_column.bucketized_column(
        i_mini_img_size, [1e-60, 2.0, 3.0, 5.0])
    linear_feature_columns.append(
        feature_column.indicator_column(i_mini_img_size))
    i_mini_img_size = feature_column.embedding_column(
        i_mini_img_size, embedding_size)
    embedding_feature_columns.append(i_mini_img_size)

    i_mini_img_size = feature_column.numeric_column('i_mini_img_size', default_value=0.0,
                                                    normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_mini_img_size)

    i_comment_count = feature_column.numeric_column(
        'i_comment_count', default_value=0.0)
    i_comment_count = feature_column.bucketized_column(i_comment_count,
                                                       [1e-60, 1.0, 2.0, 3.0, 6.0, 13.0, 26.0, 64.0, 33236.0])
    linear_feature_columns.append(
        feature_column.indicator_column(i_comment_count))
    i_comment_count = feature_column.embedding_column(
        i_comment_count, embedding_size)
    embedding_feature_columns.append(i_comment_count)

    i_comment_count = feature_column.numeric_column('i_comment_count', default_value=0.0,
                                                    normalizer_fn=lambda x: tf.log(x + sys.float_info.epsilon))
    linear_feature_columns.append(i_comment_count)

    u_brand = feature_column.categorical_column_with_vocabulary_list('u_brand',
                                                                     ['####', 'OPPO', 'HUAWEI', 'XIAOMI', 'VIVO',
                                                                      'HONOR', 'MEIZU', 'SAMSUNG', '360', 'GIONEE',
                                                                      'LEECO', 'NUBIA', 'SMARTISAN', 'LENOVO', 'ZTE',
                                                                      'COOLPAD', 'ONEPLUS', 'VOTO', '4G+', 'VIVI'],
                                                                     dtype=tf.string, default_value=0)
    linear_feature_columns.append(feature_column.indicator_column(u_brand))
    u_brand = feature_column.embedding_column(u_brand, embedding_size)
    embedding_feature_columns.append(u_brand)

    u_operator = feature_column.categorical_column_with_vocabulary_list('u_operator', ['####', 'CM', 'CT', 'CU'],
                                                                        dtype=tf.string, default_value=0)
    linear_feature_columns.append(feature_column.indicator_column(u_operator))
    u_operator = feature_column.embedding_column(u_operator, embedding_size)
    embedding_feature_columns.append(u_operator)

    u_age = feature_column.categorical_column_with_vocabulary_list('u_age',
                                                                   ['####', 'MID-AGE', 'UNKNOWN', 'YOUNG', 'ELDER',
                                                                    'YOUNGSTER', 'TEENAGE'], dtype=tf.string,
                                                                   default_value=0)
    linear_feature_columns.append(feature_column.indicator_column(u_age))
    u_age = feature_column.embedding_column(u_age, embedding_size)
    embedding_feature_columns.append(u_age)

    u_marriage = feature_column.categorical_column_with_vocabulary_list('u_marriage',
                                                                        ['####', 'UNKNOWN',
                                                                         'MARRIED', 'UNMARRIED'],
                                                                        dtype=tf.string, default_value=0)
    linear_feature_columns.append(feature_column.indicator_column(u_marriage))
    u_marriage = feature_column.embedding_column(u_marriage, embedding_size)
    embedding_feature_columns.append(u_marriage)

    u_sex = feature_column.categorical_column_with_vocabulary_list('u_sex', ['####', 'MAN', 'WOMAN', 'UNKNOWN'],
                                                                   dtype=tf.string, default_value=0)
    linear_feature_columns.append(feature_column.indicator_column(u_sex))
    u_sex = feature_column.embedding_column(u_sex, embedding_size)
    embedding_feature_columns.append(u_sex)

    u_sex_age = feature_column.categorical_column_with_vocabulary_list('u_sex_age',
                                                                       ['####', 'MAN_MID-AGE', 'MAN_UNKNOWN',
                                                                        'WOMAN_MID-AGE', 'MAN_YOUNG', 'WOMAN_YOUNG',
                                                                        'WOMAN_UNKNOWN', 'UNKNOWN_UNKNOWN', 'MAN_ELDER',
                                                                        'WOMAN_YOUNGSTER', 'MAN_YOUNGSTER',
                                                                        'WOMAN_ELDER', 'MAN_TEENAGE', 'WOMAN_TEENAGE',
                                                                        'UNKNOWN_MID-AGE', 'UNKNOWN_YOUNG',
                                                                        'UNKNOWN_YOUNGSTER'], dtype=tf.string,
                                                                       default_value=0)
    linear_feature_columns.append(feature_column.indicator_column(u_sex_age))
    u_sex_age = feature_column.embedding_column(u_sex_age, embedding_size)
    embedding_feature_columns.append(u_sex_age)

    u_sex_marriage = feature_column.categorical_column_with_vocabulary_list('u_sex_marriage',
                                                                            ['####', 'MAN_UNKNOWN', 'WOMAN_UNKNOWN',
                                                                             'MAN_UNMARRIED', 'MAN_MARRIED',
                                                                             'WOMAN_MARRIED', 'UNKNOWN_UNKNOWN',
                                                                             'WOMAN_UNMARRIED'], dtype=tf.string,
                                                                            default_value=0)
    linear_feature_columns.append(
        feature_column.indicator_column(u_sex_marriage))
    u_sex_marriage = feature_column.embedding_column(
        u_sex_marriage, embedding_size)
    embedding_feature_columns.append(u_sex_marriage)

    u_age_marriage = feature_column.categorical_column_with_vocabulary_list('u_age_marriage',
                                                                            ['####', 'MID-AGE_UNKNOWN',
                                                                             'UNKNOWN_UNKNOWN', 'YOUNG_UNKNOWN',
                                                                             'MID-AGE_MARRIED', 'YOUNG_MARRIED',
                                                                             'UNKNOWN_UNMARRIED', 'ELDER_UNKNOWN',
                                                                             'MID-AGE_UNMARRIED', 'YOUNG_UNMARRIED',
                                                                             'YOUNGSTER_UNKNOWN', 'TEENAGE_UNKNOWN',
                                                                             'ELDER_MARRIED', 'YOUNGSTER_MARRIED',
                                                                             'YOUNGSTER_UNMARRIED', 'TEENAGE_UNMARRIED',
                                                                             'ELDER_UNMARRIED', 'TEENAGE_MARRIED',
                                                                             'UNKNOWN_MARRIED'], dtype=tf.string,
                                                                            default_value=0)
    linear_feature_columns.append(
        feature_column.indicator_column(u_age_marriage))
    u_age_marriage = feature_column.embedding_column(
        u_age_marriage, embedding_size)
    embedding_feature_columns.append(u_age_marriage)

    u_activelevel = feature_column.categorical_column_with_vocabulary_list('u_activelevel',
                                                                           ['####', 'c1',
                                                                            'c2', 'c0', 'c3'],
                                                                           dtype=tf.string, default_value=0)
    linear_feature_columns.append(
        feature_column.indicator_column(u_activelevel))
    u_activelevel = feature_column.embedding_column(u_activelevel, embedding_size)
    embedding_feature_columns.append(u_activelevel)

    i_hot_news = feature_column.categorical_column_with_vocabulary_list('i_hot_news', ['####', 'c0', 'c1'],
                                                                        dtype=tf.string, default_value=0)
    linear_feature_columns.append(feature_column.indicator_column(i_hot_news))
    i_hot_news = feature_column.embedding_column(i_hot_news, embedding_size)
    embedding_feature_columns.append(i_hot_news)

    i_is_recommend = feature_column.categorical_column_with_vocabulary_list('i_is_recommend', ['####', 'c0', 'c1'],
                                                                            dtype=tf.string, default_value=0)
    linear_feature_columns.append(
        feature_column.indicator_column(i_is_recommend))
    i_is_recommend = feature_column.embedding_column(
        i_is_recommend, embedding_size)
    embedding_feature_columns.append(i_is_recommend)

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
        for i in range(len(layers)):
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
    #
    #     "i_channel": tf.placeholder(dtype=tf.string, shape=1, name='i_channel'),
    #
    #     "u_brand": tf.placeholder(dtype=tf.string, shape=1, name='u_brand'),
    #     "u_operator": tf.placeholder(dtype=tf.string, shape=1, name='u_operator'),
    #     "u_activelevel": tf.placeholder(dtype=tf.string, shape=1, name='u_activelevel'),
    #
    #     "u_age": tf.placeholder(dtype=tf.string, shape=1, name='u_age'),
    #     "u_marriage": tf.placeholder(dtype=tf.string, shape=1, name='u_marriage'),
    #     "u_sex": tf.placeholder(dtype=tf.string, shape=1, name='u_sex'),
    #     "u_sex_age": tf.placeholder(dtype=tf.string, shape=1, name='u_sex_age'),
    #     "u_sex_marriage": tf.placeholder(dtype=tf.string, shape=1, name='u_sex_marriage'),
    #     "u_age_marriage": tf.placeholder(dtype=tf.string, shape=1, name='u_age_marriage'),
    #
    #     "i_hot_news": tf.placeholder(dtype=tf.string, shape=1, name='i_hot_news'),
    #     "i_is_recommend": tf.placeholder(dtype=tf.string, shape=1, name='i_is_recommend'),
    #
    #     "i_info_exposed_amt": tf.placeholder(dtype=tf.float32, shape=1, name='i_info_exposed_amt'),
    #     "i_info_clicked_amt": tf.placeholder(dtype=tf.float32, shape=1, name='i_info_clicked_amt'),
    #     "i_info_ctr": tf.placeholder(dtype=tf.float32, shape=1, name='i_info_ctr'),
    #
    #     "i_cate_exposed_amt": tf.placeholder(dtype=tf.float32, shape=1, name='i_cate_exposed_amt'),
    #     "i_cate_clicked_amt": tf.placeholder(dtype=tf.float32, shape=1, name='i_cate_clicked_amt'),
    #     "i_category_ctr": tf.placeholder(dtype=tf.float32, shape=1, name='i_category_ctr'),
    #
    #     "c_uid_type_ctr_1": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_ctr_1'),
    #     "c_uid_type_clicked_amt_1": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_clicked_amt_1'),
    #     "c_uid_type_exposed_amt_1": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_exposed_amt_1'),
    #     "c_uid_type_ctr_3": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_ctr_3'),
    #     "c_uid_type_clicked_amt_3": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_clicked_amt_3'),
    #     "c_uid_type_exposed_amt_3": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_exposed_amt_3'),
    #     "c_uid_type_ctr_7": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_ctr_7'),
    #     "c_uid_type_clicked_amt_7": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_clicked_amt_7'),
    #     "c_uid_type_exposed_amt_7": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_exposed_amt_7'),
    #     "c_uid_type_ctr_14": tf.placeholder(dtype=tf.float32, shape=1, name='c_uid_type_ctr_14'),
    #     "c_uid_type_clicked_amt_14": tf.placeholder(dtype=tf.float32, shape=1,
    #                                                 name='c_uid_type_clicked_amt_14'),
    #     "c_uid_type_exposed_amt_14": tf.placeholder(dtype=tf.float32, shape=1,
    #                                                 name='c_uid_type_exposed_amt_14'),
    #
    #     "c_user_flavor": tf.placeholder(dtype=tf.float32, shape=1, name='c_user_flavor'),
    #
    #     "u_activetime_at1": tf.placeholder(dtype=tf.float32, shape=1, name='u_activetime_at1'),
    #     "u_activetime_at2": tf.placeholder(dtype=tf.float32, shape=1, name='u_activetime_at2'),
    #     "u_activetime_at3": tf.placeholder(dtype=tf.float32, shape=1, name='u_activetime_at3'),
    #     "u_activetime_at4": tf.placeholder(dtype=tf.float32, shape=1, name='u_activetime_at4'),
    #     "u_activetime_at5": tf.placeholder(dtype=tf.float32, shape=1, name='u_activetime_at5'),
    #
    #     "i_mini_img_size": tf.placeholder(dtype=tf.float32, shape=1, name='i_mini_img_size'),
    #     "i_comment_count": tf.placeholder(dtype=tf.float32, shape=1, name='i_comment_count'),
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
