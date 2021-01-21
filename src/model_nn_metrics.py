import datetime
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from numpy import array,asarray,zeros

import tensorflow as tf
import keras

from keras.callbacks import Callback
from keras import backend as K 
from keras.models import load_model


def f1_loss(y, y_hat):    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    f1_soft = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - f1_soft 
    
    return cost
        
    
THRESHOLD = 0.5

def f1_beta(y_true, y_pred, threshold_shift=0.5-THRESHOLD):
    beta = 2

    y_pred = K.clip(y_pred, 0, 1)

    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall) 
