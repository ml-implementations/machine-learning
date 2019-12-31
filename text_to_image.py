import glob
import os
import fire
import json
import pickle
from PIL import Image
import numpy as np
import string
from keras.utils import Sequence
from keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten, Input, Concatenate, \
    BatchNormalization, LSTM, Lambda, ReLU
from keras.models import load_model, save_model, Model
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TextToImage:
    def __init__(self):
        self.captions = dict()
        self.images = dict()

        self.width = 64
        self.height = 64
        self.max_caption_len = 128

        self.characters = [char for char in string.ascii_lowercase]
        self.digits = [char for char in string.digits]
        self.characters.extend(self.digits)
        self.characters.extend([' ', '.', ',', '-', '<EOS>', '<UNK>'])

        self.batch_size = 16
        self.epochs = 100
        self.test_count = 50
        self.gan_input = 100
        self.network = GAN_CLS(data_shape=(self.width, self.height), one_hot_dim=len(self.characters),
                               max_captions_len=self.max_caption_len, gan_input=self.gan_input)

        self.one_hot_encodings_test = None
        self.one_hot_encodings = None

    def extract_captions(self, path='data/text-to-image/text_c10'):
        for dir in os.listdir(path):
            for filename in glob.glob(os.path.join(path, dir, '*.txt')):
                with open(filename) as fp:
                    self.captions[filename.split("\\")[-1].split('.')[0]] = fp.readlines()
        json_f = json.dumps(self.captions)
        with open(path + "/captions.json", "w") as fp:
            fp.write(json_f)

    def load_captions(self, path='data/text-to-image/text_c10/captions.json'):
        with open(path, "r") as fp:
            self.captions = json.load(fp)

    def extract_images(self, path='data/text-to-image/jpg'):
        for filename in os.listdir(path):
            im = Image.open(os.path.join(path, filename))
            im.resize((self.width, self.height), Image.BILINEAR)
            image = np.asarray(im)
            self.images[filename.split('.')[0]] = image

        with open('/'.join(path.split('/')[:-1]) + "/images.pkl", "wb") as fp:
            pickle.dump(self.images, fp)

    def load_images(self, path='data/text-to-image/images.pkl'):
        with open(path, "rb") as fp:
            self.images = pickle.load(fp)

    def one_hot_encoding(self, path='data/text-to-image/text_c10'):
        self.one_hot_encodings = {}

        self.load_captions()

        for key, caption in self.captions.items():
            caption = caption[0].replace("\n", "")
            caption_chars = [char for char in caption]

            one_hot_vec = np.zeros(shape=(self.max_caption_len, len(self.characters)))

            for i, char in enumerate(caption_chars):
                if i >= self.max_caption_len:
                    continue
                try:
                    one_hot_vec[i, self.characters.index(char)] = 1
                except ValueError:
                    one_hot_vec[i, self.characters.index("<UNK>")] = 1

            for i in range(self.max_caption_len - len(caption_chars)):
                one_hot_vec[i + len(caption_chars), self.characters.index("<EOS>")] = 1

            one_hot_vec[-1, self.characters.index("<EOS>")] = 1
            self.one_hot_encodings[key] = one_hot_vec

        with open(path + "/one_hot_captions.pkl", "wb") as fp:
            pickle.dump(self.one_hot_encodings, fp)

    def load_one_hot_encodings(self, path='data/text-to-image/text_c10'):
        with open(path + "/one_hot_captions.pkl", "rb") as fp:
            self.one_hot_encodings = pickle.load(fp)
            self.one_hot_encodings_test = {k: self.one_hot_encodings[k]
                                           for k in list(self.one_hot_encodings)[:self.test_count]}

    def train(self):
        self.load_captions()
        self.load_one_hot_encodings()

        data_generator = DataGenerator(self.one_hot_encodings, self.batch_size,
                                       images_path='data/text-to-image/jpg', shuffle=False,
                                       image_resize_shape=(self.width, self.height))

        for i in range(self.epochs):
            for j, (images, one_hot_captions) in enumerate(data_generator):
                noise = np.random.normal(0, 1, (self.batch_size, 100))
                minibatch_y = np.ones(self.batch_size) - 0.01
                generated_x = self.network.generator.predict([noise, one_hot_captions])
                generated_y = np.zeros(self.batch_size)

                minibatch_y = np.expand_dims(minibatch_y, axis=-1)
                generated_y = np.expand_dims(generated_y, axis=-1)

                self.network.discriminator.trainable = True

                disc_noise = np.random.normal(scale=0.2, size=(self.batch_size, self.width, self.height, 3))
                disc_loss_real = self.network.discriminator.train_on_batch(
                    x=[images + disc_noise, one_hot_captions], y=[minibatch_y])
                disc_loss_fake = self.network.discriminator.train_on_batch(
                    x=[generated_x + disc_noise, one_hot_captions], y=[generated_y])

                noise = np.random.normal(0, 1, (self.batch_size, self.gan_input))
                gan_y = np.ones(self.batch_size)

                gen_loss = self.network.gan.train_on_batch([noise, one_hot_captions], gan_y)

                print("Batch: {}".format(j+1))
                print("Discriminator Loss: ", disc_loss_real[0] + disc_loss_fake[0])
                print("Gen Loss: ", gen_loss)

            self.sample_gan(i)
            print(self.network.generator)
            print("+++++++Epoch: {}++++++++".format(i + 1))

    def sample_gan(self, epoch):
        noise = np.random.normal(0, 1, (1, self.gan_input))
        test_caption = np.random.randint(self.test_count)
        test_key = list(self.one_hot_encodings_test.keys())[test_caption]
        img = self.network.generator.predict([noise,
                                             np.expand_dims(self.one_hot_encodings_test[test_key], axis=0)])
        print(img)
        print(self.captions[test_key][0])
        img = np.squeeze(img, axis=0)
        img = (img + 1) * 255.0
        plt.imshow(img)
        plt.show()


class DataGenerator(Sequence):
    def __init__(self, one_hot_encodings, batch_size, images_path, shuffle, image_resize_shape):
        self.one_hot_encodings = one_hot_encodings
        self.batch_size = batch_size
        self.images_path = images_path
        self.image_resize_shape = image_resize_shape
        self.shuffle = shuffle
        self.list_IDs = os.listdir(self.images_path)[:1600]
        self.indexes = None

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size, *self.image_resize_shape, 3))
        y = []

        for i, id in enumerate(list_IDs_temp):
            im = Image.open(os.path.join(self.images_path, id))
            im = im.resize((self.image_resize_shape[0], self.image_resize_shape[1]), Image.BILINEAR)
            x[i, :, :, :] = np.asarray(im)
            y.append(self.one_hot_encodings[id.split(".")[0]])

        x = 2 * (x - np.amin(x) / (np.amax(x) - np.amin(x))) - 1

        return x, np.array(y)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


class GAN_CLS:
    def __init__(self, data_shape, one_hot_dim, max_captions_len, gan_input):
        self.generator = None
        self.discriminator = None
        self.gan = None

        self.data_shape = data_shape

        self.gan_input = gan_input
        self.embedding_dim = 256
        self.compressed_dim = 128
        self.one_hot_dim = one_hot_dim
        self.max_captions_len = max_captions_len

        self.create_generator()
        self.create_discriminator()
        self.create_gan()

    def create_discriminator(self):
        input_im = Input(shape=(*self.data_shape, 3))
        input_one_hot = Input(shape=(self.max_captions_len, self.one_hot_dim))

        embeddings = LSTM(self.embedding_dim)(input_one_hot)

        dense_em_1 = Dense(self.compressed_dim)(embeddings)
        dense_em_1 = LeakyReLU(alpha=0.2)(dense_em_1)
        dense_em_1 = Reshape((4, 4, -1))(dense_em_1)

        x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(input_im)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(64, (4, 4), strides=(4, 4), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Concatenate(axis=-1)([x, dense_em_1])

        x = Conv2D(16, (4, 4), padding='valid')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, activation='sigmoid')(x)

        x = Lambda(lambda y: K.squeeze(y, -1))(x)
        x = Lambda(lambda y: K.squeeze(y, -1))(x)

        model = Model(inputs=[input_im, input_one_hot], outputs=x)

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()

        self.discriminator = model

    def create_generator(self):
        input_z = Input(shape=(self.gan_input, ))
        input_one_hot = Input(shape=(self.max_captions_len, self.one_hot_dim))

        embeddings = LSTM(self.embedding_dim)(input_one_hot)

        dense_em_1 = Dense(self.compressed_dim)(embeddings)
        dense_em_1 = LeakyReLU(alpha=0.2)(dense_em_1)

        x = Concatenate(axis=1)([input_z, dense_em_1])

        n_nodes = 128 * 2 * 2
        x = Dense(n_nodes)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((2, 2, 128))(x)
        x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)

        x = Conv2D(3, (3, 3),  activation="tanh", padding='same')(x)

        model = Model(inputs=[input_z, input_one_hot], outputs=x)

        model.summary()

        self.generator = model

    def create_gan(self):
        self.discriminator.trainable = False

        input_z = Input(shape=(self.gan_input,))
        input_one_hot = Input(shape=(None, self.one_hot_dim))

        im = self.generator([input_z, input_one_hot])
        out = self.discriminator([im, input_one_hot])

        opt = Adam(lr=0.0002, beta_1=0.5)
        model = Model(inputs=[input_z, input_one_hot], outputs=out)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        self.gan = model


if __name__ == '__main__':
    fire.Fire(TextToImage)
