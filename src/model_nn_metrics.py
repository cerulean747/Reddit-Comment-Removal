import datetime
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()

        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average=None)
        _val_recall = recall_score(val_targ, val_predict,average=None)
        _val_precision = precision_score(val_targ, val_predict,average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f'val_f1: {_val_f1}  val_precision: {_val_precision}  val_recall: {_val_recall}')
        (_val_f1, _val_precision, _val_recall)
        return


class SaveEveryEpoch(Callback):
    def __init__(self, model_name, *args, **kwargs):
        self.model_checkpoint_paths = []
        self.model_name = model_name
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs):
        # I suppose here it's a Functional model
        print(logs['acc'])
        path_to_checkpoint = (
            str(datetime.datetime.now()).split(' ')[0] 
            + f'_{self.model_name}'
            + f'_{epoch:02d}.hdf5'
        )
        self.model.save(path_to_checkpoint)
        self.model_checkpoint_paths.append(path_to_checkpoint)

        
THRESHOLD = 0.5

def f1(y_true, y_pred, threshold_shift=0.5-THRESHOLD):
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
