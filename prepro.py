import numpy as np
import json
import os
import collections
import random
from tqdm import tqdm

from keras.preprocessing import image as Image
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import text as Text
import utils


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

    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]}

    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    caption_data['annotations'] = ([annotation for annotation in caption_data['annotations']
                                    if len(Text.text_to_word_sequence(annotation['caption'])) <= max_length])
    caption_list = []
    for annotation in caption_data['annotations']:
        caption_list.append(annotation['caption'])

    tokenizer, vocab = _create_tokenizer_and_vocab(caption_list)

    # dataset is a dictionary, filename : info, info is also a dict that has two keys 'features' and 'captions'
    # 'captions' is a ndarray, with shape [num_captions,17]
    # 'features' is a array with shape [1,4096]

    print('building dataset ....')

    train_dataset = build_dataset(caption_data['annotations'], image_dir, feature_model, id_to_filename,
                                  vocab, tokenizer, max_length)

    total_data_len = len(train_dataset)
    val_len = int(total_data_len*0.2)

    print('building validation set ....')

    val_dataset = dict()
    for _ in tqdm(range(val_len)):
        key = random.choice(list(train_dataset.keys()))
        val_dataset[key] = train_dataset[key]
        del train_dataset[key]

    print('training set num = %d' % len(train_dataset))
    print('validation set num = %d' % len(val_dataset))

    return train_dataset, val_dataset, vocab


def build_dataset(data_annotations, image_dir, feature_model, id_to_filename, vocab, tokenizer, max_length):

    dataset = dict()

    for annotation in tqdm(data_annotations):
        image_id = annotation['image_id']
        filename = id_to_filename[image_id]
        if filename not in dataset:
            this_file = dict()
            this_file['feature'] = feature_extraction(feature_model, os.path.join(image_dir, filename))
            this_file['caption'] = text_to_vec(annotation['caption'], tokenizer, vocab, max_length)
            dataset[filename] = this_file
        else:
            text_vec = text_to_vec(annotation['caption'], tokenizer, vocab, max_length)
            dataset[filename]['caption'] = np.vstack([dataset[filename]['caption'], text_vec])

    return dataset


def text_to_vec(text, tokenizer, vocab, max_length):

    caption_idx = tokenizer.texts_to_sequences([text])

    text_vec = list()
    text_vec.append(vocab['<START>'])
    text_vec.extend(caption_idx[0])
    text_vec.append(vocab['<END>'])
    while len(text_vec) < max_length + 2:
        text_vec.append(vocab['<NULL>'])

    return np.reshape(text_vec, [1, -1])


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


def main():

    # caption_list = collect_coco_cations_list()
    caption_file = '/media/VSlab3/fionakuo/CV_FINAL/coco/captions_train2017.json'
    image_dir = '/media/VSlab3/fionakuo/CV_FINAL/coco_image/train2017'

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    feature_model = feature_extractor('VGG19')
    train_dataset, val_dataset, vocab = _process_caption_data(caption_file, image_dir, feature_model)

    utils.save_pickle(train_dataset, '/media/VSlab3/fionakuo/CV_FINAL/data/train_dict.pkl')
    utils.save_pickle(val_dataset, '/media/VSlab3/fionakuo/CV_FINAL/data/val_dict.pkl')
    utils.save_pickle(vocab, '/media/VSlab3/fionakuo/CV_FINAL/data/vocab.pkl')
    """
    All data is ordered from annotation
    from caption array to image file : annotations[row_idx][file_name]
    feature array is the same 
    """


if __name__ == '__main__':
    main()
