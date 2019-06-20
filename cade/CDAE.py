from tensorflow.python.keras.regularizers import l2
import tensorflow as tf


def create(I, U, K, hidden_activation, output_activation, q=0.5, l=0.01):
    '''
    create model
    Reference:
      Yao Wu, Christopher DuBois, Alice X. Zheng, Martin Ester.
        Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
          The 9th ACM International Conference on Web Search and Data Mining (WSDM'16), p153--162, 2016.

    :param I: number of items
    :param U: number of users
    :param K: number of units in hidden layer
    :param hidden_activation: activation function of hidden layer
    :param output_activation: activation function of output layer
    :param q: drop probability
    :param l: regularization parameter of L2 regularization
    :return: CDAE
    :rtype: keras.models.Model
    '''

    x_item = tf.keras.Input(shape=(I,), name='x_item')
    h_item = tf.keras.layers.Dropout(rate=q)(x_item)
    h_item = tf.keras.layers.Dense(
        units=K, kernel_regularizer=l2(l), bias_regularizer=l2(l))(h_item)

    x_user = tf.keras.Input(shape=(1,), dtype='int32', name='x_user')
    h_user = tf.keras.layers.Embedding(
        input_dim=U, output_dim=K, input_length=1, embeddings_regularizer=l2(l))(x_user)
    h_user = tf.keras.layers.Flatten()(h_user)

    h = tf.keras.layers.Add()([h_item, h_user])
    if hidden_activation:
        h = tf.keras.layers.Activation(hidden_activation)(h)
    y = tf.keras.layers.Dense(units=I, activation=output_activation)(h)
    model = tf.keras.Model(inputs=[x_item, x_user], outputs=y)
    return model
