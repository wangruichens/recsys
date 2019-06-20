# Only build_raw_serving_input_receiver_fn support using REST api
# Otherwise need example.SerializeToString()

import json
import requests
import tensorflow as tf

# tf.enable_eager_execution()
#
# record_iterator = tf.python_io.tf_record_iterator(path='part-r-00000')
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#     # print(example)
#     break

p = {"c_uid_type_clicked_amt_1": 107.0,
     "c_uid_type_clicked_amt_14": 517.0,
     "c_uid_type_clicked_amt_3": 517.0,
     "c_uid_type_clicked_amt_7": 517.0,
     "c_uid_type_ctr_1": 0.53233802318573,
     "c_uid_type_ctr_14": 0.5340909957885742,
     "c_uid_type_ctr_3": 0.5340909957885742,
     "c_uid_type_ctr_7": 0.5340909957885742,
     "c_uid_type_exposed_amt_1": 201.0,
     "c_uid_type_exposed_amt_14": 968.0,
     "c_uid_type_exposed_amt_3": 968.0,
     "c_uid_type_exposed_amt_7": 968.0,
     "c_user_flavor": 1.0312000513076782,
     "i_cate_clicked_amt": 490384.0,
     "i_cate_exposed_amt": 3666584.0,
     "i_category_ctr": 0.1337440013885498,
     "i_channel": "xinwen",
     "i_comment_count": 0.0,
     "i_hot_news": "####",
     "i_id": 459924,
     "i_info_clicked_amt": 6.0,
     "i_info_ctr": 0.125,
     "i_info_exposed_amt": 48.0,
     "i_is_recommend": "####",
     "i_mini_img_size": 0.0,
     "u_activelevel": "####",
     "u_activetime_at1": 0.0,
     "u_activetime_at2": 0.0,
     "u_activetime_at3": 0.0,
     "u_activetime_at4": 0.0,
     "u_activetime_at5": 0.0,
     "u_age": "####",
     "u_age_marriage": "####",
     "u_brand": "####",
     "u_id": 1044329,
     "u_marriage": "####",
     "u_operator": "####",
     "u_sex": "####",
     "u_sex_age": "####",
     "u_sex_marriage": "####"
     }


# saved_model_cli run --dir /home/wangrc/Downloads/export/1560231853/ --tag_set serve --signature_def serving_default --input_exprs='examples=[b"\n\x8d\t\n\x16\n\ti_channel\x12\t\n\x07\n\x05meinv\n\x1e\n\x12i_info_exposed_amt\x12\x08\x12\x06\n\x04\x00\x00@B\n\x1e\n\x12i_info_clicked_amt\x12\x08\x12\x06\n\x04\x00\x00\xc0@\n\x16\n\ni_info_ctr\x12\x08\x12\x06\n\x04\x00\x00\x00>\n\x1e\n\x12i_cate_exposed_amt\x12\x08\x12\x06\n\x04`\xca_J\n\x1e\n\x12i_cate_clicked_amt\x12\x08\x12\x06\n\x04\x00r\xefH\n\x1a\n\x0ei_category_ctr\x12\x08\x12\x06\n\x040\xf4\x08>\n\x1c\n\x10c_uid_type_ctr_1\x12\x08\x12\x06\n\x04NG\x08?\n$\n\x18c_uid_type_clicked_amt_1\x12\x08\x12\x06\n\x04\x00\x00\xd6B\n$\n\x18c_uid_type_exposed_amt_1\x12\x08\x12\x06\n\x04\x00\x00IC\n\x1c\n\x10c_uid_type_ctr_3\x12\x08\x12\x06\n\x040\xba\x08?\n$\n\x18c_uid_type_clicked_amt_3\x12\x08\x12\x06\n\x04\x00@\x01D\n$\n\x18c_uid_type_exposed_amt_3\x12\x08\x12\x06\n\x04\x00\x00rD\n\x1c\n\x10c_uid_type_ctr_7\x12\x08\x12\x06\n\x040\xba\x08?\n$\n\x18c_uid_type_clicked_amt_7\x12\x08\x12\x06\n\x04\x00@\x01D\n$\n\x18c_uid_type_exposed_amt_7\x12\x08\x12\x06\n\x04\x00\x00rD\n\x1d\n\x11c_uid_type_ctr_14\x12\x08\x12\x06\n\x040\xba\x08?\n%\n\x19c_uid_type_clicked_amt_14\x12\x08\x12\x06\n\x04\x00@\x01D\n%\n\x19c_uid_type_exposed_amt_14\x12\x08\x12\x06\n\x04\x00\x00rD\n\x19\n\rc_user_flavor\x12\x08\x12\x06\n\x04]\xfe\x83?\n\x16\n\nu_operator\x12\x08\n\x06\n\x04####\n\x1c\n\x10u_activetime_at1\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at2\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at3\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at4\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at5\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1b\n\x0fi_mini_img_size\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1b\n\x0fi_comment_count\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x13\n\x07u_brand\x12\x08\n\x06\n\x04####\n\x19\n\ru_activelevel\x12\x08\n\x06\n\x04####\n\x11\n\x05u_age\x12\x08\n\x06\n\x04####\n\x16\n\nu_marriage\x12\x08\n\x06\n\x04####\n\x11\n\x05u_sex\x12\x08\n\x06\n\x04####\n\x15\n\tu_sex_age\x12\x08\n\x06\n\x04####\n\x1a\n\x0eu_sex_marriage\x12\x08\n\x06\n\x04####\n\x1a\n\x0eu_age_marriage\x12\x08\n\x06\n\x04####\n\x16\n\ni_hot_news\x12\x08\n\x06\n\x04####\n\x1a\n\x0ei_is_recommend\x12\x08\n\x06\n\x04####\n\x0f\n\x04u_id\x12\x07\x1a\x05\n\x03\xe9\xde?\n\x0f\n\x04i_id\x12\x07\x1a\x05\n\x03\x94\x89\x1c"]'

# tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=deepfm \
#   --model_base_path="/home/wangrc/Desktop/"

data = json.dumps({"signature_name": "serving_default", "instances": [p]})

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/deepfm:predict', data=data, headers=headers)
print(json_response.text)
predictions = json.loads(json_response.text)['predictions']
print(predictions)
