#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:48:08 2020

@author: sapna
"""
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Dropout, Lambda, Add, Reshape 
from keras_vggface.vggface import VGGFace
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np

class NormL(Layer):
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel',
                                 shape=(1, input_shape[-1]),
                                 initializer='ones',
                                 trainable=True)
        self.b = self.add_weight(name='kernel',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out * self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape
        
class MacularNet():
    def __init__(self):
        self.hidden_dim = 512
        self.nb_classes = 3
        
    def MultiHeadsAttModel(self, l=8 * 8, d=512, dv=64, dout=512, nv=8):
        v1 = Input(shape=(l, d))
        q1 = Input(shape=(l, d))
        k1 = Input(shape=(l, d))

        v2 = Dense(dv * nv, activation="relu")(v1)
        q2 = Dense(dv * nv, activation="relu")(q1)
        k2 = Dense(dv * nv, activation="relu")(k1)

        v = Reshape([l, nv, dv])(v2)
        q = Reshape([l, nv, dv])(q2)
        k = Reshape([l, nv, dv])(k2)

        att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv),
                     output_shape=(l, nv, nv))([q, k])  # l, nv, nv
        att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)
        out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 3]), output_shape=(l, nv, dv))([att, v])
        out = Reshape([l, d])(out)
        out = Add()([out, q1])
        out = Dense(dout, activation="relu")(out)
        return Model(inputs=[q1, k1, v1], outputs=out)

    def model(self):
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        vgg_out = vgg_model.get_layer('conv5_3').output
        x = Reshape([14 * 14, 512])(vgg_out)
        att_mod = self.MultiHeadsAttModel(l=14 * 14, d=512, dv=8 * 8, dout=32, nv=8)
        x = att_mod([x, x, x])
        x = Reshape([14, 14, 32])(x)
        x = NormL()(x)
        x = Flatten()(x)
        x = Dense(self.hidden_dim, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.hidden_dim, activation='relu', name='fc7')(x)
        x = Dropout(0.5)(x)
        out = Dense(self.nb_class, activation='softmax', name='fc8')(x)
        model = Model(vgg_model.input, out)
        return model