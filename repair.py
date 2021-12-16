from logging import exception
import keras
import sys
import h5py
from keras.utils.generic_utils import validate_config
from tensorflow.keras import models
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
weights_filename = "models/bd_weights.h5"
model_filename = "models/bd_net.h5"
clean_data_filename = "data/cl/valid.h5"
clean_test_data_filename = "data/cl/test.h5"
save_model_filename = "models/repaired_bd_net2.h5"
save_weights_filename = "models/repaired_bd_weights2.h5"
poisoned_data_filename = "data/bd/bd_test.h5"
unars =0
def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    return x_data/255, y_data

def unrepair():
    cl_x_test, cl_y_test = data_loader(clean_test_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)
    bd_model = keras.models.load_model(model_filename)

    cl_label_p = np.argmax(bd_model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    print('Clean Classification accuracy:', clean_accuracy)
    bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
    asr = 100-np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print('Attack Success Rate:', asr)
    return asr


def repair(model, num_nu):
    x_valid,y_valid = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)
    weights_dict = {}
    bias_dict = {}
    acc_dict = {}
    x = keras.Input(shape=(55, 47, 3), name='input')
    conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1')(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
    conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2')(pool_1)
    pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
    conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3')(pool_2)
    reModel = keras.Model(inputs=x, outputs=conv_3)
    model.load_weights(weights_filename)
    for i in range(1, 4):
        layer = "conv_" + str(i)
        weights, bias = model.get_layer(layer).get_weights()
        reModel.get_layer(layer).set_weights((weights, bias))
        weights_dict[layer] = weights
        bias_dict[layer] = bias
    yhat = reModel.predict(x_valid)
    out = np.mean(yhat, axis=0)
    score = np.argsort(np.sum(out, axis=(0, 1)))
    yclean = np.argmax(model.predict(x_valid), axis=1)
    acc = np.mean(np.equal(yclean, y_valid)) * 100
    weights, bias = weights_dict['conv_3'], bias_dict['conv_3']
    for i in range (1,num_nu):
        weights[:, :, :, score[i]] = np.zeros(np.shape(weights[:, :, :, score[i]]))
        yrepair = np.argmax(model.predict(x_valid), axis=1)
        acc = np.mean(np.equal(yrepair, y_valid)) * 100
        acc_dict[i] = acc
        bias[score[i]] = 0
        model.get_layer('conv_3').set_weights((weights, bias))
    model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(x_valid, y_valid, epochs=5)
    yrepair = np.argmax(model.predict(x_valid), axis=1)
    acc = np.mean(np.equal(yrepair, y_valid)) * 100
    print("Repaired Classification accuracy(%): ", acc)
    bd_label_p = np.argmax(model.predict(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print('Attack Success Rate After Repaired:', asr)
    #print('Attack Success Rate Reduced (%):', unars - asr)
    return model

if __name__ == "__main__":
    # define input
    x = keras.Input(shape=(55, 47, 3), name='input')
    # feature extraction
    conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1')(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
    conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2')(pool_1)
    pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
    conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3')(pool_2)
    pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(conv_3)
    # first interpretation model
    flat_1 = keras.layers.Flatten()(pool_3)
    fc_1 = keras.layers.Dense(160, name='fc_1')(flat_1)
    # second interpretation model
    conv_4 = keras.layers.Conv2D(80, (2, 2), activation='relu', name='conv_4')(pool_3)
    flat_2 = keras.layers.Flatten()(conv_4)
    fc_2 = keras.layers.Dense(160, name='fc_2')(flat_2)
    # merge interpretation
    merge = keras.layers.Add()([fc_1, fc_2])
    add_1 = keras.layers.Activation('relu')(merge)
    drop = keras.layers.Dropout(0.5)
    # output
    y_hat = keras.layers.Dense(1283, activation='softmax', name='output')(add_1)
    model = keras.Model(inputs=x, outputs=y_hat)
    # summarize layers
    # print(model.summary())
    # plot graph
    # plot_model(model, to_file='model_architecture.png')
    unars = unrepair()
    reModel = repair(model, 30)
    reModel.save(save_model_filename)
    reModel.save_weights(save_weights_filename)