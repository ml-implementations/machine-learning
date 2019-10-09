import numpy as np
import fire
import cv2
import matplotlib.pyplot as plt
from keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten
from keras.models import load_model, save_model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adam


class GAN:
    def __init__(self):
        self.discriminator = None
        self.generator = None
        self.gan = None
        self.gan_input = 100
        self.data_shape = (28, 28, 1)
        self.X_train = None
        self.X_test = None
        self.batch_size = 32
        self.epochs = 100
        self.generator_model_path = 'models/gan.hdf5'
        self.test_count = 9

    def create_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=self.data_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.summary()

        self.discriminator = model

    def create_generator(self):
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, input_dim=self.gan_input))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))

        self.generator = model

    def create_gan(self):
        # make weights in the discriminator not trainable
        self.discriminator.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(self.generator)
        # add the discriminator
        model.add(self.discriminator)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        self.gan = model

    def load_data(self):
        (self.X_train, labels), (self.X_test, _) = mnist.load_data()
        self.X_train = self.X_train.astype(np.float32) / 255.0

    def train(self):
        self.load_data()
        self.create_discriminator()
        self.create_generator()
        self.create_gan()

        for i in range(self.epochs):
            for k in range(int(self.X_train.shape[0] / self.batch_size)):
                noise = np.random.normal(0, 1, (self.batch_size, 100))
                minibatch_x = self.X_train[k * self.batch_size:(k + 1) * self.batch_size]
                minibatch_x = np.expand_dims(minibatch_x, axis=-1)
                minibatch_y = np.ones(self.batch_size) - 0.01
                generated_x = self.generator.predict(noise)
                generated_y = np.zeros(self.batch_size)

                minibatch_y = np.expand_dims(minibatch_y, axis=-1)
                generated_y = np.expand_dims(generated_y, axis=-1)

                self.discriminator.trainable = True
                self.discriminator.train_on_batch(minibatch_x, minibatch_y)
                self.discriminator.train_on_batch(generated_x, generated_y)

                # noise = self.generate_latent_points(100, self.batch_size)
                noise = np.random.normal(0, 1, (self.batch_size, 100))
                gan_y = np.ones(self.batch_size)

                self.gan.train_on_batch(noise, gan_y)

            if i % 5 == 0:
                self.sample_gan(i)
                save_model(self.generator, self.generator_model_path)

            print("Epoch: ", i + 1)

    def sample_gan(self, epoch):
        noise = np.random.normal(0, 1, (1, 100))
        img = self.generator.predict(noise)
        img = np.squeeze(img, axis=0)
        img = np.squeeze(img, axis=-1)
        img = img * 255.0
        cv2.imwrite('gan_generated/img_{}.png'.format(epoch), img)

    def plot_results(self, generated):
        fig = plt.figure(figsize=(28, 28))
        columns = np.sqrt(self.test_count)
        rows = np.sqrt(self.test_count)
        for i in range(1, int(columns) * int(rows)):
            fig.add_subplot(rows, columns, i)
            plt.imshow(generated[i], cmap='gray_r')
        plt.show()

    def test(self):
        generated = []
        for i in range(self.test_count):
            noise = np.random.normal(0, 1, (1, 100))
            img = self.generator.predict(noise)
            img = np.squeeze(img, axis=0)
            img = np.squeeze(img, axis=-1)
            generated.append(img * 255.0)
        self.plot_results(generated)


if __name__ == "__main__":
    fire.Fire(GAN)
