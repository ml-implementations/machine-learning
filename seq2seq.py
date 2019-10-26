"""
Implementation of a Sequence-to-Sequence model for English to German translation in PyTorch
"""

import numpy as np
import fire
import torch
from torch.optim import Adam
from multiprocessing import set_start_method

torch.set_default_tensor_type(torch.cuda.FloatTensor)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class Seq2Seq:
    def __init__(self):
        self.model = Model()
        self.corpus_en = None
        self.corpus_de = None
        self.vocab_en = None
        self.vocab_de = None
        self.vocab_len_en = None
        self.vocab_len_de = None
        self.data_en_path = "./data/en_de/train_en.dat"
        self.data_de_path = "./data/en_de/train_de.dat"
        self.embedding_dim = 256
        self.hidden_dim = 256
        self.model_name = "./models/seq2seq.h5"

        # model
        self.batch_size = 32
        self.model_loss = torch.nn.CrossEntropyLoss()
        self.model_optim = None
        self.model_optim = Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.max_len = None

    def load_dataset(self, path):
        with open(path) as fp:
            corpus = fp.readlines()
            vocab = list(set(" ".join(corpus).split(" ")))
            vocab.extend(["<BLANK>", "<EOS>"])
            vocab_len = len(vocab)

        len_corpus = len(max(corpus, key=len)) + 1
        if self.max_len is None:
            self.max_len = len_corpus
        if self.max_len < len_corpus:
            self.max_len = len_corpus

        return corpus, vocab, vocab_len

    def preprocess_corpus(self, corpus, lang, padding, eos):
        corpus_encoded = np.ones(shape=(len(corpus), self.max_len), dtype=np.float32) * padding
        for i, sentence in enumerate(corpus):
            for j, word in enumerate(sentence.split(" ")):
                corpus_encoded[i, j] = self.word_vocab_encode(word, lang)
            corpus_encoded[i, len(sentence.split(" "))] = eos

        return corpus_encoded

    def word_vocab_encode(self, word, lang):
        if lang == "en":
            return self.vocab_en.index(word)
        else:
            return self.vocab_de.index(word)

    def save_preprocessed_corpus(self):
        self.corpus_en, self.vocab_en, self.vocab_len_en = self.load_dataset(self.data_en_path)
        self.corpus_de, self.vocab_de, self.vocab_len_de = self.load_dataset(self.data_de_path)

        self.corpus_en = self.preprocess_corpus(self.corpus_en, "en", self.vocab_len_en - 2, self.vocab_len_en - 1)
        self.corpus_de = self.preprocess_corpus(self.corpus_de, "de", self.vocab_len_de - 2, self.vocab_len_de - 1)

        np.save('./data/en_de/corpus_en', self.corpus_en)
        np.save('./data/en_de/corpus_de', self.corpus_de)

    def train(self):
        _, self.vocab_en, self.vocab_len_en = self.load_dataset(self.data_en_path)
        _, self.vocab_de, self.vocab_len_de = self.load_dataset(self.data_de_path)

        self.corpus_en = torch.tensor(np.load('./data/en_de/corpus_en.npy')).long()
        self.corpus_de = torch.tensor(np.load('./data/en_de/corpus_de.npy')).long()

        self.model_optim.zero_grad()
        out = self.model(self.corpus_en[:self.batch_size], self.corpus_de[:self.batch_size])
        loss = self.model_loss(out, self.corpus_de[:self.batch_size])
        print("Loss: ", loss.item())
        loss.backward()
        self.model_optim.step()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(8120)
        self.decoder = Decoder(10161)

    def forward(self, x, y):
        x, state = self.encoder(x)
        x = self.decoder(y, state)

        return x


class Encoder(torch.nn.Module):
    def __init__(self, vocab_len_en):
        super().__init__()
        self.embedding1 = torch.nn.Embedding(vocab_len_en, 256)
        self.lstm1 = torch.nn.LSTM(256, hidden_size=256, num_layers=2)

    def forward(self, x):
        x = self.embedding1(x)
        x, state = self.lstm1(x)

        return x, state


class Decoder(torch.nn.Module):
    def __init__(self, vocab_len_de):
        super().__init__()
        self.embedding1 = torch.nn.Embedding(num_embeddings=vocab_len_de, embedding_dim=256)
        self.lstm1 = torch.nn.LSTM(input_size=256, hidden_size=256)
        self.fc1 = torch.nn.Linear(in_features=256, out_features=vocab_len_de)

    def forward(self, x, state):
        x = self.embedding1(x)
        # print(state[0].view(2, 1, -1, 256).shape[1])
        x, _ = self.lstm1(x, (state[0].view(2, 1, -1, 256)[1], state[1].view(2, 1, -1, 256)[1]))
        x = torch.softmax(self.fc1(x), dim=1)

        return x


def main():
    fire.Fire(Seq2Seq)


if __name__ == "__main__":
    main()
