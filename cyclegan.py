import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers import RandomNormal
from keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, Input, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow_datasets as tfds
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

# implementation of cyclegan in keras
# https://github.com/EvolvedSquid/tutorials/tree/master/cyclegan

# Load dataset
data, metadata = tfds.load('cycle_gan/monet2photo', with_info=True, as_supervised=True)
train_x, train_y, test_x, test_y = data['trainA'], data['trainB'], data['testA'], data['testB']

# Settings
epochs = 50
LAMBDA = 10

img_rows, img_cols, channels = 256, 256, 3
weight_initializer = RandomNormal(stddev=0.02)


def preprocess_image(image, _):
    return tf.reshape(tf.cast(tf.image.resize(image, (int(img_rows), int(img_cols))), tf.float32) / 127.5 - 1,
                      (1, img_rows, img_cols, channels))


train_x = train_x.map(preprocess_image)
train_y = train_y.map(preprocess_image)
test_x = test_x.map(preprocess_image)
test_y = test_y.map(preprocess_image)


def Ck(input, k, use_instancenorm=True):
    block = Conv2D(k, (4, 4), strides=2, padding='same', kernel_initializer=weight_initializer)(input)
    if use_instancenorm:
        block = InstanceNormalization(axis=-1)(block)
    block = LeakyReLU(0.2)(block)

    return block


def discriminator():
    dis_input = Input(shape=(img_rows, img_cols, channels))

    d = Ck(dis_input, 64, False)
    d = Ck(d, 128)
    d = Ck(d, 256)
    d = Ck(d, 512)

    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=weight_initializer)(d)

    model = Model(inputs=dis_input, outputs=d)
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss=binary_crossentropy)

    return model


def dk(k, use_instancenorm=True):
    block = Sequential()
    block.add(Conv2D(k, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer))
    if use_instancenorm:
        block.add(InstanceNormalization(axis=-1))
    block.add(Activation('relu'))

    return block


def uk(k):
    block = Sequential()
    block.add(Conv2DTranspose(k, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer))
    block.add(InstanceNormalization(axis=-1))
    block.add(Activation('relu'))

    return block


def generator_loss(gen_input, gen_validity, real_x, real_y, cyc_x, cyc_y, gen_out):
    gen_adv_loss = binary_crossentropy(K.ones_like(gen_validity), gen_validity)
    cyc_x_loss = K.mean(K.abs(real_x - cyc_x))
    cyc_y_loss = K.mean(K.abs(real_y - cyc_y))

    id_loss = K.mean(K.abs(gen_input - gen_out))

    return K.mean(gen_adv_loss + (cyc_x_loss + cyc_y_loss) * LAMBDA + id_loss * 0.5 * LAMBDA)


def generator():
    gen_input = Input(shape=(img_rows, img_cols, channels))
    gen_validity = Input(shape=(16, 16, 1))
    cyc_x = Input(shape=(img_rows, img_cols, channels))
    cyc_y = Input(shape=(img_rows, img_cols, channels))
    real_x = Input(shape=(img_rows, img_cols, channels))
    real_y = Input(shape=(img_rows, img_cols, channels))

    # Layers for the encoder part of the model
    encoder_layers = [
        dk(64, False),
        dk(128),
        dk(256),
        dk(512),
        dk(512),
        dk(512),
        dk(512),
        dk(512)
    ]

    # Layers for the decoder part of the model
    decoder_layers = [
        uk(512),
        uk(512),
        uk(512),
        uk(512),
        uk(256),
        uk(128),
        uk(64)
    ]

    gen = gen_input

    # Add all the encoder layers, and keep track of them for skip connections
    skips = []
    for layer in encoder_layers:
        gen = layer(gen)
        skips.append(gen)

    skips = skips[::-1][1:]  # Reverse for looping and get rid of the layer that directly connects to decoder

    # Add all the decoder layers and skip connections
    for skip_layer, layer in zip(skips, decoder_layers):
        gen = layer(gen)
        gen = Concatenate()([gen, skip_layer])

    # Final layer
    gen = Conv2DTranspose(channels, (3, 3), strides=2, padding='same', kernel_initializer=weight_initializer,
                          activation='tanh')(gen)

    model = Model(inputs=[gen_input, gen_validity, real_x, real_y, cyc_x, cyc_y], outputs=gen)
    model.add_loss(generator_loss(gen_input, gen_validity, real_x, real_y, cyc_x, cyc_y, gen))
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss=None)

    # Compose model
    return model


def generate_images(x, y, fake_x, fake_y):
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(x * 0.5 + 0.5)
    axes[0, 0].axis('off')
    axes[1, 1].imshow(fake_x * 0.5 + 0.5)
    axes[1, 1].axis('off')
    axes[0, 1].imshow(y * 0.5 + 0.5)
    axes[0, 1].axis('off')
    axes[1, 0].imshow(fake_y * 0.5 + 0.5)
    axes[1, 0].axis('off')
    plt.tight_layout()
    plt.show()


# Define the models
generator_g = generator()
generator_f = generator()

discriminator_x = discriminator()
discriminator_y = discriminator()


# Manually loop through epochs
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))

    # Each batch
    for k, (batch_real_x, batch_real_y) in enumerate(tfds.as_numpy(tf.data.Dataset.zip((train_x, train_y)))):
        print("Batch no: ", k)
        if k == 10: break

        fake_y = generator_g.predict([batch_real_x, np.zeros(shape=(1, 16, 16, 1)),
                                     np.zeros(shape=(1, img_rows, img_cols, channels)),
                                     np.zeros(shape=(1, img_rows, img_cols, channels)),
                                     np.zeros(shape=(1, img_rows, img_cols, channels)),
                                     np.zeros(shape=(1, img_rows, img_cols, channels))], steps=1)

        discriminator_y.trainable = True
        discriminator_y.train_on_batch(x=batch_real_y, y=np.ones(shape=(1, 16, 16, 1)))
        discriminator_y.train_on_batch(x=fake_y, y=np.zeros(shape=(1, 16, 16, 1)))

        fake_x = generator_f.predict([batch_real_y, np.zeros(shape=(1, 16, 16, 1)),
                                      np.zeros(shape=(1, img_rows, img_cols, channels)),
                                      np.zeros(shape=(1, img_rows, img_cols, channels)),
                                      np.zeros(shape=(1, img_rows, img_cols, channels)),
                                      np.zeros(shape=(1, img_rows, img_cols, channels))], steps=1)
        discriminator_x.trainable = True
        discriminator_x.train_on_batch(x=batch_real_x, y=np.ones(shape=(1, 16, 16, 1)))
        discriminator_x.train_on_batch(x=fake_x, y=np.zeros(shape=(1, 16, 16, 1)))

        discriminator_x.trainable = False
        discriminator_y.trainable = False

        gen_g_validity = discriminator_y.predict(fake_y)
        generator_g.train_on_batch(x=[batch_real_x, gen_g_validity, batch_real_x, batch_real_y, fake_x, fake_y],
                                   y=None)

        gen_f_validity = discriminator_x.predict(fake_x)
        generator_f.train_on_batch(x=[batch_real_y, gen_f_validity, batch_real_x, batch_real_y, fake_x, fake_y],
                                   y=None)

        if k % 10 == 0:
            generate_images(np.squeeze(batch_real_x),
                            np.squeeze(batch_real_y),
                            np.squeeze(fake_x),
                            np.squeeze(fake_y))
