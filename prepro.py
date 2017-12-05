import tensorflow as tf
import numpy as np
import json
import os
import argparse
import collections

from keras.preprocessing import image as Image
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import text as Text
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help='image_dir', type=str,
                    default='/media/VSlab3/fionakuo/CV_FINAL/coco_image/train2017')
parser.add_argument('--train_caption_file', help='train_caption_file', type=str,
                    default='/media/VSlab3/fionakuo/CV_FINAL/coco/captions_train2017.json')
parser.add_argument('--output_dir', help='output_dir', type=str, default='/media/VSlab3/fionakuo/CV_FINAL/data')
args = parser.parse_args()


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


def _process_caption_data(caption_file, image_dir, feature_model, max_length=15):
    data = collections.namedtuple('Caption', 'annotation, captions, features, vocab')
    with open(caption_file) as f:
        caption_data = json.load(f)
    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    caption_data['annotations'] = ([annotation for annotation in caption_data['annotations']
                                    if len(Text.text_to_word_sequence(annotation['caption'])) <= max_length])

    caption_list = []

    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        caption_list.append(annotation['caption'])

    tokenizer, vocab = _create_tokenizer_and_vocab(caption_list)

    captions = _build_text_vec(caption_list, vocab, tokenizer, max_length)

    features = _build_feature_vec(caption_data['annotations'], feature_model)

    assert captions.shape[0] == features.shape[0]

    return data(
        annotation=caption_data['annotations'],
        captions=captions,
        features=features
    ), vocab


def _create_tokenizer_and_vocab(texts):
    """
    :param texts: list of text
    :return: a tokenizer

    Use tokenizer.word_index to see the vocabulary
    """
    tokenizer = Text.Tokenizer()

    tokenizer.fit_on_texts(texts)

    for key in tokenizer.word_index.keys():
        tokenizer.word_index[key] += 2
    vocab = {'<NULL>': 0, '<START>': 1, '<END>': 2}
    vocab.update(tokenizer.word_index)
    return tokenizer, vocab


def _build_text_vec(caption_list, vocab, tokenizer, max_length=15):

    caption_idx_list = tokenizer.texts_to_sequences(caption_list)
    # Make fixed-length vector

    captions = []
    for i, caption_idx in enumerate(caption_idx_list, 0):
        text_vec = list()
        text_vec.append(vocab['<START>'])
        text_vec.extend(caption_idx)
        text_vec.append(vocab['<END>'])
        while len(text_vec) < max_length+2:
            text_vec.append(vocab['<NULL>'])
        captions.append(text_vec)
    captions = np.vstack(np.asarray(captions))
    return captions


def _build_feature_vec(annotations, model):
    feature_vec = np.zeros([len(annotations), 4096])
    for i, annotation in enumerate(annotations):
        image_path = annotation['file_name']
        feature_vec[i, :] = feature_extraction(model, image_path)
        print('Process caption:', i)
    return feature_vec


def main():

    caption_file = args.caption_file
    image_dir = args.image_dir

    feature_model = feature_extractor('VGG19')
    data = _process_caption_data(caption_file, image_dir, feature_model)._asdict()

    utils.save_pickle(data, '/media/VSlab3/fionakuo/CV_FINAL/data/train_data.pkl')
    utils.save_pickle(data['vocab'], 'media/VSlab3/fionakuo/CV_FINAL/data/vocab.pkl')

    """
    All data is ordered from annotation
    from caption array to image file : annotations[row_idx][file_name]
    feature array is the same 
    """


if __name__ == '__main__':

    main()
