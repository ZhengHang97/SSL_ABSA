import keras
from keras.models import *
from keras.layers import *

import tensorflow as tf

# print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]


class AddBeta(Layer):
    def __init__(self, **kwargs):
        super(AddBeta, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.built:
            return

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)

        self.built = True
        super(AddBeta, self).build(input_shape)

    def call(self, x, training=None):
        return tf.add(x, self.beta)


class G_Guass(Layer):
    def __init__(self, **kwargs):
        super(G_Guass, self).__init__(**kwargs)

    def wi(self, init, name):
        if init == 1:
            return self.add_weight(name='guess_' + name,
                                   shape=(self.size,),
                                   initializer='ones',
                                   trainable=True)
        elif init == 0:
            return self.add_weight(name='guess_' + name,
                                   shape=(self.size,),
                                   initializer='zeros',
                                   trainable=True)
        else:
            raise ValueError("Invalid argument '%d' provided for init in G_Gauss layer" % init)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[0][-1]

        init_values = [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]
        self.a = [self.wi(v, 'a' + str(i + 1)) for i, v in enumerate(init_values)]
        super(G_Guass, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        z_c, u = x

        def compute(y):
            return y[0] * tf.sigmoid(y[1] * u + y[2]) + y[3] * u + y[4]

        mu = compute(self.a[:5])
        v = compute(self.a[5:])

        z_est = (z_c - mu) * v + mu
        return z_est

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.size)


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


def add_noise(inputs, noise_std):
    return Lambda(lambda x: x + tf.keras.backend.random_normal(tf.shape(x)) * noise_std)(inputs)


def LD(layer_sizes=[],  # 768,768,300,6
                          noise_std=0.3,
                          denoising_cost=[1000.0, 10.0, 0.10, 0.10]):  # , 0.10, 0.10, 0.10
    L = len(layer_sizes)-1  # number of layers:3

    #  task1
    inputs_l = Input(shape=(layer_sizes[0],))  # ??????????????????768??????
    inputs_u = Input(shape=(layer_sizes[0],))


    fc_enc = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[1:]]  # 768,300,6
    fc_dec = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[:-1]] # 768 768 300
    betas = [AddBeta() for l in range(L)]  # l = 0,1,2

    def encoder(inputs, noise_std):
        h = add_noise(inputs, noise_std)
        all_z = [None for _ in range(len(layer_sizes))]  # ???????????????z
        all_z[0] = h  # z0=x+noise

        for l in range(1, L + 1):  # l=1,2,3
            z_pre = fc_enc[l - 1](h)
            z = Lambda(batch_normalization)(z_pre)
            z = add_noise(z, noise_std)

            if l == L:
                h = Activation('softmax')(betas[l - 1](z))
            else:
                h = Activation('relu')(betas[l - 1](z))

            all_z[l] = z

        return h, all_z

    y_c_l, _ = encoder(inputs_l, noise_std)
    y_l, _ = encoder(inputs_l, 0.0)

    y_c_u, corr_z = encoder(inputs_u, noise_std)
    y_u, clean_z = encoder(inputs_u, 0.0)

    # Decoder
    d_cost = []  # to store the denoising cost of all layers
    for l in range(L, -1, -1):  # ???3?????????0??????????????????1,???3,2,1,0???
        z, z_c = clean_z[l], corr_z[l]
        if l == L:
            u = y_c_u
        else:
            u = fc_dec[l](z_est)
        u = Lambda(batch_normalization)(u)
        z_est = G_Guass()([z_c, u])
        d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est - z), 1)) / layer_sizes[l]) * denoising_cost[l])

    u_cost = tf.add_n(d_cost)  # denoising cost
    y_c_l = Lambda(lambda x: x[0])([y_c_l, y_l, y_c_u, y_u, u, z_est, z])
    cost = -tf.reduce_mean(tf.reduce_sum(y_l * tf.math.log(y_c_l), 1))  # supervised cost


    loss = u_cost + cost
    tr_m = Model(inputs=[inputs_l, inputs_u], outputs=y_c_l)
    '''
    loss = u_cost
    tr_m = Model([inputs_l, inputs_u], y_c_l)
    '''

    tr_m.add_loss(loss)
    tr_m.compile(keras.optimizers.Adam(0.01), loss='categorical_crossentropy', metrics=METRICS)  # metrics=['accuracy']

    tr_m.metrics_names.append("den_loss")
    tr_m.metrics.append(loss)


    te_m = Model(inputs_l, y_l)
    tr_m.test_model = te_m


    return tr_m



