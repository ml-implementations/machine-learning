import glob
import os
import fire
import json
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class TextToImage:
    def __init__(self):
        self.captions = dict()
        self.images = dict()
        self.width = 228
        self.height = 228

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


if __name__ == '__main__':
    fire.Fire(TextToImage)
