import pickle
import collections
import os
import json
import tensorflow as tf
import numpy as np
import math
from keras.preprocessing import image as Image
from keras.applications.vgg19 import VGG19
from keras.models import Model

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
        print('Saved %s..' % path)


def _dict2namedtuple(dictionary):

    return collections.namedtuple('GerericDic', dictionary.keys())(**dictionary)


def load_coco(data_dir, mode):
    if mode == 'train':
        data = _dict2namedtuple(load_pickle(os.path.join(data_dir, 'train_data.pkl')))
    elif mode == 'val':
        data = _dict2namedtuple(load_pickle(os.path.join(data_dir, 'val_data.pkl')))
    else:
        raise ValueError('mode is not provided')

    return data

def feature_extractor(model_type=None):

    if model_type == 'VGG19':
        base_model = VGG19(weights='imagenet')
        model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc2').output)
    else:
        raise ValueError('model type is not provided')

    return model


def feature_extraction(model, image_path=None):

    assert isinstance(image_path, str), 'Image is not provided'

    img = Image.load_img(image_path, target_size=(224, 224))
    img = Image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    features = model.predict(img)

    return features

"""
def _process_caption(caption_in, caption_out, feature):

    caption_in = caption_in[:-2]
    caption_out = caption_out[1:]

    return caption_in, caption_out, feature


def data_loader(data_dir, mode, batch_size=16, shuffle=False, num_epochs=1):

    Data = collections.namedtuple('data', 'inputs, targets, features')
    if mode == 'train':
        data = _dict2namedtuple(load_pickle(os.path.join(data_dir, 'train_data.pkl')))
    elif mode == 'val':
        data = _dict2namedtuple(load_pickle(os.path.join(data_dir, 'val_data.pkl')))
    else:
        raise ValueError('mode is not provided')
    captions_tensor = tf.placeholder(dtype=tf.int32, shape=[None, None])
    features_tensor = tf.placeholder(dtype=tf.float32, shape=[None, None])

    dataset = tf.contrib.data.Dataset.from_tensor_slices((captions_tensor, captions_tensor, features_tensor))
    dataset = dataset.map(_process_caption, num_parallel_calls=5)
    dataset = dataset.shuffle(buffer_size=10000) if shuffle else dataset
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    caption_in, caption_out, features = iterator.get_next()



    return Data(
        inputs=caption_in,
        targets=caption_out,
        features=features
    )
"""
