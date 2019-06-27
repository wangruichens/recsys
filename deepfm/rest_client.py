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

p = { "i_id": 459924,
     "u_id": 1044329,
     }

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
