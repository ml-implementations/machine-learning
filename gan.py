import numpy as np
import fire
import cv2
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras.models import Model, load_model, save_model
from keras.datasets import mnist


class GAN:
    def __init__(self):
        self.discriminator = None
        self.generator = None
        self.gan = None
        self.gan_input = (100, )
        self.data_shape = (28, 28)
        self.generator_input = self.gan_input
        self.generator_output = (np.prod(self.data_shape), )
        self.X_train = None
        self.X_test = None
        self.batch_size = 32
        self.epochs = 100
        self.generator_model_path = 'models/gan.hdf5'
        self.test_count = 9

    def create_discriminator(self):
        input = Input(self.generator_output)

        x = Dense(1024)(input)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.discriminator = Model(input=input, output=output)
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam')

    def create_generator(self):
        input = Input(self.gan_input)

        x = Dense(256)(input)
        x = LeakyReLU(0.2)(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(0.2)(x)

        output = Dense(self.generator_output[0], activation='tanh')(x)

        self.generator = Model(input=input, output=output)
        self.generator.compile(loss='binary_crossentropy', optimizer='adam')

    def create_gan(self):
        input = Input(self.gan_input)

        self.discriminator.trainable = False
        x = self.generator(input)
        output = self.discriminator(x)

        self.gan = Model(input=input, output=output)
        self.gan.compile(loss='binary_crossentropy', optimizer='adam')

    def load_data(self):
        (self.X_train, _), (self.X_test, _) = mnist.load_data()
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        print(self.X_train.shape)

    def train(self):
        self.load_data()
        self.create_discriminator()
        self.create_generator()
        self.create_gan()

        for i in range(self.epochs):
            for k in range(int(self.X_train.shape[0]/self.batch_size)):
                noise = np.random.normal(0, 1, (self.batch_size, 100))
                minibatch_x = self.X_train[k*self.batch_size:(k+1)*self.batch_size]
                minibatch_x = np.reshape(minibatch_x, (self.batch_size, 784))
                minibatch_y = np.ones(self.batch_size) - 0.01
                generated_x = self.generator.predict(noise)
                generated_y = np.zeros(self.batch_size)

                self.discriminator.trainable = True
                self.discriminator.train_on_batch(minibatch_x, minibatch_y)
                self.discriminator.train_on_batch(generated_x, generated_y)

                noise = np.random.normal(0, 1, (self.batch_size, 100, ))
                gan_y = np.ones(self.batch_size)

                self.gan.train_on_batch(noise, gan_y)

            if i % 5 == 0:
                self.sample_gan(i)
                save_model(self.generator, self.generator_model_path)

            print("Epoch: ", i+1)

    def sample_gan(self, epoch):
        noise = np.random.normal(0, 1, (1, 100))
        img = self.generator.predict(noise)
        img = np.reshape(img, (28, 28))
        img = img * 255
        cv2.imwrite('gan_generated/img_{}.png'.format(epoch), img)

    def plot_results(self, generated):
        fig = plt.figure(figsize=(28, 28))
        columns = np.sqrt(self.test_count)
        rows = np.sqrt(self.test_count)
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(generated[i])
        plt.show()

    def test(self):
        generator = load_model(self.generator_model_path)
        generated = []
        for _ in range(self.test_count):
            noise = np.random.normal(0, 1, (1, 100))
            generated.append(generator.predict(noise))

        self.plot_results(generated)


if __name__ == "__main__":
    fire.Fire(GAN)
