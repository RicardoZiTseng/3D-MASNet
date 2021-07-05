import os
from functools import partial

import keras
import numpy as np
import tensorflow as tf
from keras.backend import clear_session, set_session
from keras.optimizers import Adam

from models.dunet import DUNetMixACB, cross_entropy
from util.data import DataGen
from util.prep import maybe_mkdir_p
from util.misc import save_params, cosine_annealing

def main(params):
    # ======================================
    #       Set Environment Variable      #
    # ======================================
    work_path = os.path.abspath('./workdir')
    save_path = os.path.join(work_path, 'save', params.name)
    maybe_mkdir_p(save_path)
    save_file_name = os.path.join(
        save_path, '{epoch:02d}.h5')
    save_params(params, os.path.join(save_path, 'train.json'))

    # ======================================
    #       Close Useless Information     #
    # ======================================
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    clear_session()

    # ==============================================================
    #         Set GPU Environment And Initialize Networks         #
    # ==============================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    model = DUNetMixACB().build_model()

    if params.load_model_file is not None:
        print('loadding ', params.load_model_file)
        model.load_weights(params.load_model_file)

    save_schedule = keras.callbacks.ModelCheckpoint(filepath=save_file_name, period=10)

    # ===================================================================
    #        Set Training Callbacks and Initialize Data Generator      #
    # ===================================================================
    train_gen = DataGen(params.train_dir, params.train_ids).make_gen()

    lr_schedule_fn = partial(cosine_annealing, lr_init=params.lr_init, lr_min=params.lr_min)

    lr_schedule = keras.callbacks.LearningRateScheduler(lr_schedule_fn)

    call_backs = [lr_schedule, save_schedule]
    
    model.compile(optimizer=Adam(lr=params.lr_init), loss=cross_entropy)
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=params.train_nums_per_epoch//params.batch_size,
                                  epochs=params.epochs, callbacks=call_backs)

