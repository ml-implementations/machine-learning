import fire
import umap
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1.0, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)

        dot_product = tf.matmul(
            feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
        )

        logits = tf.divide(
            dot_product, self.temperature
        )

        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class SupervisedContrastiveLearner:

    def __init__(self):
        self.encoder = None
        self.epochs = 1
        self.batch_size = 16
        self.num_classes = 10
        self.input_shape = (32, 32, 3)
        self.embedding_dim = 128
        self.temperature = 0.05
        self.dropout = 0.2
        self.lr = 0.01

        self.encoder_path = "./models/supervised_contrastive_encoder"

        self.train_data = None
        self.test_data = None

    def load_data(self):
        self.train_data, self.test_data = keras.datasets.cifar10.load_data()
        print(f"Data loaded. Train shape: {self.train_data[0].shape}, "
              f"Test shape: {self.test_data[0].shape}")

    def create_encoder(self):
        spine = keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=self.input_shape, pooling="avg"
        )

        inputs = keras.Input(shape=self.input_shape)
        features = spine(inputs)
        outputs = keras.layers.Dense(self.embedding_dim, activation="relu")(features)
        model = keras.Model(inputs=inputs, outputs=outputs, name="supervised_contrastive_encoder")

        return model

    def train(self):
        # Load data
        self.load_data()

        # Create encoder
        encoder = self.create_encoder()
        encoder.summary()

        # Compile encoder
        encoder.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SupervisedContrastiveLoss(self.temperature),
        )

        # Train encoder
        x_train, y_train = self.train_data[0], self.train_data[1]
        # keras.backend.clear_session()
        encoder.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs)

        # Save model
        encoder.save(self.encoder_path)

    def visualize_embeddings(self):
        # Load data
        self.load_data()

        # Load model
        encoder = keras.models.load_model(self.encoder_path, compile=False)
        encoder.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SupervisedContrastiveLoss(self.temperature),
        )

        # Compute embeddings
        x, y = self.test_data[0], self.test_data[1]
        embeddings = encoder.predict(x)
        print(f"Encoder embedding shape: {embeddings.shape}")

        # UMAP
        reducer = umap.UMAP()
        umap_embeddings = reducer.fit_transform(embeddings)
        print(f"UMAP embedding shape: {umap_embeddings.shape}")

        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=y)
        plt.title("UMAP for CIFAR-10")
        plt.show()


if __name__ == "__main__":
    fire.Fire(SupervisedContrastiveLearner)
