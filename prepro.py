import numpy as np
import json
import os
import argparse

from keras.preprocessing import image as Image
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import text as Text
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', help='image_dir', type=str,
                    default='./data/images')
parser.add_argument('--captions_dir', help='caption directory', type=str,
                    default='./data/annotations')
parser.add_argument('--output_dir', help='output_dir', type=str, default='./data/data')
args = parser.parse_args()


def _process_caption_data(caption_dir, image_dir, feature_model, max_length=15):
    train_captions = os.path.join(caption_dir, 'captions_train2017.json')
    train_images_dir = os.path.join(image_dir, 'train2017')
    val_captions = os.path.join(caption_dir, 'captions_val2017.json')
    val_images_dir = os.path.join(image_dir, 'val2017')

    train_captions_data, train_captions_list = _create_data_caption_list(train_captions, train_images_dir, max_length)
    val_captions_data, val_captions_list = _create_data_caption_list(val_captions, val_images_dir, max_length)

    tokenizer, vocab = _create_tokenizer_and_vocab(train_captions_list)
    
    train_captions = _build_text_vec(train_captions_list, vocab, tokenizer, max_length)
    train_features = _build_feature_vec(train_captions_data['annotations'], feature_model)

    train_data = dict()
    train_data['annotation'] = train_captions_data['annotations']
    train_data['captions'] = train_captions
    train_data['features'] = train_features
    
    val_captions = _build_text_vec(val_captions_list, vocab, tokenizer, max_length)
    val_features = _build_feature_vec(val_captions_data['annotations'], feature_model)

    val_data = dict()
    val_data['annotation'] = val_captions_data['annotations']
    val_data['captions'] = val_captions
    val_data['features'] = val_features

    return train_data, val_data, vocab


def _create_data_caption_list(caption_file, images_dir, max_length=15):
    with open(caption_file) as f:
        captions_data = json.load(f)
    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] for image in captions_data['images']}

    captions_data['annotations'] = ([annotation for annotation in captions_data['annotations']
                                    if len(Text.text_to_word_sequence(annotation['caption'])) <= max_length])
    captions_list = []

    for annotation in captions_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(images_dir, id_to_filename[image_id])
        captions_list.append(annotation['caption'])

    return captions_data, captions_list


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
        feature_vec[i, :] = utils.feature_extraction(model, image_path)
        print('Process caption:', i)
    return feature_vec


def main():

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    feature_model = utils.feature_extractor('VGG19')
    train_data, val_data, vocab = _process_caption_data(args.captions_dir, args.images_dir, feature_model)

    utils.save_pickle(train_data, os.path.join(args.output_dir, 'train_data.pkl'))
    utils.save_pickle(val_data, os.path.join(args.output_dir, 'val_data.pkl'))
    utils.save_pickle(vocab, os.path.join(args.output_dir, 'vocab.pkl'))

    """
    All data is ordered from annotation
    from caption array to image file : annotations[row_idx][file_name]
    feature array is the same 
    """


if __name__ == '__main__':
    main()
