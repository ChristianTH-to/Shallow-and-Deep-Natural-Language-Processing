# Attention based, bidirectional long short term memory recurrent neural network

# Dependencies

import warnings
warnings.filterwarnings("ignore")


# Data

import pandas as pd
dataset = pd.read_csv('bt_data_train_set_1_5.csv').fillna('')


# Label each user 0: bt_1 and 1: bt_5

from sklearn import preprocessing

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(dataset.Name.values)

# Split data into xtrain/ytrain xval/yval sets

from sklearn.model_selection import train_test_split
xtrain, xval, ytrain, yval = train_test_split(dataset.Message.values, y, 
                                                  stratify=y, 
                                                  random_state=10, 
                                                  test_size=0.1, shuffle=True)


# Computing the weight of each class to balance the prediction

from sklearn.utils import class_weight

bt_class_weights = y
values = class_weight.compute_class_weight('balanced',bt_class_weights,y)
class_weights = dict(zip(bt_class_weights, values))

print(class_weights)


# Import GloVe embeddings

from tqdm import tqdm
import numpy as np

embedding_signal = {}
e = open('glove.840B.300d.word2vec.txt') # Need the full representation which includes stopwords

for p in tqdm(e):
    real_num = p.split(' ')
    word = real_num[0]
    coefs = np.asarray(real_num[1:], dtype='float32')
    embedding_signal[word] = coefs
e.close()

print('Found %s word vectors.' % len(embedding_signal)) # Returns embedding progress bar


# Transform each users name vector into 1 unique classes for all observations 

import keras
from keras.utils import np_utils

ytrain_enc = np_utils.to_categorical(ytrain)
yval_enc = np_utils.to_categorical(yval)

#from sklearn.preprocessing import StandardScaler
#sc_y = StandardScaler()
#ytrain_enc = sc_y.fit_transform(ytrain_enc)
#yval_enc = sc_y.fit_transform(yval_enc)

# Tokenize text 

from keras.preprocessing import sequence, text

token = text.Tokenizer(num_words=2196017)
max_len = 30

# Transforms tokenized text to sequence of ints

token.fit_on_texts(list(xtrain) + list(xval))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xval)

# Zero pad the sequences and scales x

from keras.preprocessing.sequence import pad_sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xval_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

#sc_X = StandardScaler()
#xtrain_pad = sc_X.fit_transform(xtrain_pad)
#xval_pad = sc_X.fit_transform(xval_pad)
#from sklearn.preprocessing import normalize
#train_pad = normalize(xtrain_pad)
#xval_pad = normalize(xval_pad)

word_index = token.word_index

# Create an embedding matrix for the words we have in the dataset

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in tqdm(word_index.items()):
    embedding_vector = embedding_signal.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Attention Layer   

from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer    
from keras import backend as K

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):

        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
    

# Computing the weight of each class to balance the prediction

from sklearn.utils import class_weight

bt_class_weights = y
values = class_weight.compute_class_weight('balanced',bt_class_weights,y)
class_weights = dict(zip(bt_class_weights, values))

print(class_weights)


# Tensorflow metric wrappers for keras called at .compile

import tensorflow as tf
import functools

def as_keras_metric(method):
    @functools.wraps(method)

    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([update_op]):
            value = tf.identity(value)

        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)
auc = as_keras_metric(tf.metrics.auc)


# Defines f2 score

def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # Implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)



def tp_score(y_true, y_pred, threshold=0.1):
    tp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))

    return tp


def fp_score(y_true, y_pred, threshold=0.1):
    fp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(K.abs(y_true - K.ones_like(y_true)))), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=-1
    )

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    return fp


def fn_score(y_true, y_pred, threshold=0.1):
    fn_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.abs(K.cast(K.greater(y_pred, K.constant(threshold)),
            	'float') - K.ones_like(y_pred)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    return fn

# Defines precision curve

def precision_score(y_true, y_pred, threshold=0.1):
    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)

    return tp / (tp + fp)

# Defines recall curve

def recall_score(y_true, y_pred, threshold=0.1):
    tp = tp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    return tp / (tp + fn)

# Defines f1 score

def f_score(y_true, y_pred, threshold=0.1, beta=2):
    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1+beta**2) * ((precision * recall) / ((beta**2)*precision + recall))


# LSTM with glove embedding layer, one bidirectional lstm layer, one attention layer and one dense layer
# Tried a convolutional layer to speed up training. testing on several epochs, did not speed up training

# Dependencies

from keras.layers import Flatten, Bidirectional, SpatialDropout1D, GlobalMaxPool1D
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import metrics
from time import time

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len, # I think input length describes the input length of the matrix to the model
                     trainable=False)) # Freezes out word embedding parameters. 
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(150, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
#model.add(GlobalMaxPool1D())
model.add(Attention(max_len))
model.add(Dense(50)) #, activation='relu'))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.9))
model.add(Dense(2))
model.add(Activation('sigmoid')) # Softmax 
model.compile(keras.optimizers.SGD(lr=1e-4, decay=1e-5),
              loss='binary_crossentropy',
              metrics = ['acc', auc, recall, precision, f2_score, f_score]) 
model.summary()
# Stops the model before epoch threshold of 80
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose='auto') 

# Oversampling
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=12, ratio=1.0)
#xtrain_pad, ytrain_enc = sm.fit_sample(xtrain_pad,ytrain_enc)


# During training, enables the model to treat class 1 as important as class 0

class_weight = {0:1,
                1:2} # [0.6470374744715928]

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#metrics = Metrics()

# Fit data to the model and initialize training

history = model.fit(xtrain_pad, y=ytrain_enc, batch_size=1024, epochs=1, verbose=1, validation_data=(xval_pad, yval_enc), 
                    shuffle=True,callbacks=[tensorboard], class_weight=class_weight)

# Plots score of loss, accuracy, recall, precision, auroc, f1 & f2 metrics

print(history.history['loss'])
print(history.history['val_loss']) 
print(history.history['acc'])
print(history.history['val_acc'])
print(history.history['recall'])
print(history.history['val_recall'])
print(history.history['precision'])
print(history.history['val_precision'])
print(history.history['auc'])
print(history.history['val_auc'])
print(history.history['f2_score'])
print(history.history['val_f2_score'])
print(history.history['f_score'])
print(history.history['val_f_score'])
print(history.history['ck'])
print(history.history['val_ck'])


# Lstm loss plot

from matplotlib.pyplot import figure
import matplotlib.pyplot as pyplot
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('train loss vs validation loss') 
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc = 'upper right') 
pyplot.show()


# Lstm acc plot

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('train accuracy vs validation accuracy') 
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc = 'lower right') 
pyplot.show()


# Lstm auroc plot

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pyplot.plot(history.history['auc'],color='red', marker = '.')
pyplot.plot([0,140],[0,1], linestyle='--')
#pyplot.plot(history.history['val_auc'])
pyplot.title('auroc') 
pyplot.ylabel('roc')
pyplot.xlabel('auc')
pyplot.legend(['roc','auc'], loc = 'lower right') 
pyplot.show()


# Lstm recall plot

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pyplot.plot(history.history['recall'], color='orange', marker = '.')
#pyplot.plot([0,1],[0.5,0.5], linestyle='--')
pyplot.title('recall curve') 
pyplot.ylabel('recall')
pyplot.xlabel('threshold')
#pyplot.legend(['recall','precision'], loc = 'lower right') 
pyplot.show()


# Lstm precision plot

%matplotlib inline
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pyplot.plot(history.history['precision'], color='magenta', marker = '.')
#pyplot.plot([0,1],[0.5,0.5], linestyle='--')
pyplot.title('auroc') 
pyplot.ylabel('precision')
pyplot.xlabel('threshold')
#pyplot.legend(['precision','recall'], loc = 'lower right') 
pyplot.show()


# Tensorflow computed f2_score plot

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pyplot.plot(history.history['val_f2_score'], color='red', marker = '.')
#pyplot.plot([0,1],[0.5,0.5], linestyle='--')
pyplot.plot([0,120],[0,1], linestyle='--')
pyplot.title('auroc') 
pyplot.ylabel('roc')
pyplot.xlabel('auc')
pyplot.legend(['roc','auc'], loc = 'lower right') 
pyplot.show()


# Tensorflow computed f_score plot

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pyplot.plot(history.history['f_score'], color='purple', marker = '.')
#pyplot.plot([0,1],[0.5,0.5], linestyle='--')
pyplot.title('f score curve') 
pyplot.ylabel('precision')
pyplot.xlabel('recall')
#pyplot.legend(['precision','recall'], loc = 'lower right') 
pyplot.show()


# Model evaluation on the test set

df_test = pd.read_csv('bt_1_bt_5_groundtruth_test.csv')
x_test = df_test['Message'].fillna("_na_").values
x_test = token.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=max_len)
y_test = model.predict([x_test], batch_size=1024, verbose=1) #1024
y_test = (y_test > 0.5).astype(int)
print(y_test)