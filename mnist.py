import urllib.request
import os
import gzip
import struct
import numpy as np

# Code graciously adapted from @alenic:
# https://alenic.github.io/snippets/2021/02/08/mnist-loader-python.html

BASE_URL = "http://yann.lecun.com/exdb/mnist/"
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def get_train(data_dir='~/mnist/'):
    train_image_path = _download_file(data_dir, TRAIN_IMAGES)
    train_label_path = _download_file(data_dir, TRAIN_LABELS)
    train_images = parse_images(train_image_path)
    train_labels = parse_labels(train_label_path)
    return train_images, train_labels


def get_test(data_dir='~/mnist/'):
    test_image_path = _download_file(data_dir, TEST_IMAGES)
    test_label_path = _download_file(data_dir, TEST_LABELS)
    test_images = parse_images(test_image_path)
    test_labels = parse_labels(test_label_path)
    return test_images, test_labels


def _download_file(data_dir, url_suffix):
    url = BASE_URL + url_suffix
    output_path = data_dir + url_suffix
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(output_path):
        print("Downloading", url, "to", output_path, "...")
        urllib.request.urlretrieve(url, output_path)
    else:
        print(output_path, "already downloaded.")
    return output_path


def parse_images(f):
    images = []
    with gzip.open(f, 'rb') as fp:
        header = struct.unpack('>4i', fp.read(16))
        magic, size, width, height = header

        if magic != 2051:
            raise RuntimeError("'%s' is not an MNIST image set." % f)

        chunk = width * height
        for _ in range(size):
            img = struct.unpack('>%dB' % chunk, fp.read(chunk))
            img_np = np.array(img, np.uint8)
            images.append(img_np)
    return np.array(images) / 255


def parse_labels(f):
    with gzip.open(f, 'rb') as fp:
        header = struct.unpack('>2i', fp.read(8))
        magic, size = header

        if magic != 2049:
            raise RuntimeError("'%s' is not an MNIST label set." % f)

        labels = struct.unpack('>%dB' % size, fp.read())
    return np.array(labels, np.int32)