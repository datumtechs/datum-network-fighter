"""
Defines the actual model for making policy and value predictions given an observation.
"""

import hashlib
import json
import os
from logging import getLogger
import multiprocessing as mp

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from xiangqi import N_COLS, N_ROWS
from inferencer import ChessModelAPI
from data_helper import download_model_data, read_content, upload_data, zip_and_b64encode

# noinspection PyPep8Naming

logger = getLogger(__name__)


class NNModel:
    def __init__(self, config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.api = None
        self.graph = None
        self.session = None
        self.lock = mp.Lock()

    def get_pipes(self, num=1):
        """
        Creates a list of pipes on which observations of the game state will be listened for. Whenever
        an observation comes in, returns policy and value network predictions on that pipe.

        :param int num: number of pipes to create
        :return str(Connection): a list of all connections to the pipes that were created
        """
        if self.api is None:
            self.api = ChessModelAPI(self)
            self.api.start()
        return [self.api.create_pipe() for _ in range(num)]

    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        self.graph = tf.get_default_graph()
        self.session = tf.Session(graph=self.graph)
        K.set_session(self.session)

        mc = self.config.model
        in_x = x = Input((14, N_ROWS, N_COLS))

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-" + str(mc.cnn_first_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax",
                           name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x, index):
        mc = self.config.model
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv1-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv2-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path, io_channel):
        """

        :param str config_path: path to the file containing the entire configuration
        :param str weight_path: path to the file containing the model weights
        :return: true iff successful in loading
        """
        mc = self.config.model
        resources = self.config.resource
        logger.info(f"model config path {config_path}, weight path {os.path.abspath(weight_path)}, resources {resources}")
        if config_path == resources.model_best_config_path:
            ret = download_model_data(self.config, io_channel)
            if not ret:
                return False

        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.info(f"loading model from {config_path}")
            with self.lock:
                with open(config_path, "rt") as f:
                    self.model = Model.from_config(json.load(f))
                self.model.load_weights(weight_path)
                self.model._make_predict_function()
                self.digest = self.fetch_digest(weight_path)
                logger.info(f"loaded model digest = {self.digest}")
                return True
        else:
            logger.info(f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path, io_channel, upload=False):
        """

        :param str config_path: path to save the entire configuration to
        :param str weight_path: path to save the model weights to
        """
        logger.info(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        # self.digest = self.fetch_digest(weight_path)
        # logger.debug(f"saved model digest {self.digest}")

        mc = self.config.model
        resources = self.config.resource
        if upload:  # mc.distributed and config_path == resources.model_best_config_path:
            dir_ = os.path.basename(os.path.dirname(config_path))
            assert ' ' not in dir_
            data = read_content(config_path, text=True)
            upload_data(data, self.config, io_channel, f'upload_model_cfg{dir_} ')
            data = read_content(weight_path, text=False)
            data1 = zip_and_b64encode(data)
            upload_data(data1, self.config, io_channel, f'upload_model_weight{dir_} ')
