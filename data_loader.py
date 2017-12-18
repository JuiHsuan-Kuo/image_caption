import random
import utils
import numpy as np
import collections
import tensorflow as tf
from tqdm import tqdm


class Sampler:
    def __init__(self, data, batch_size=128):
        self.data = data
        self.batch_size = batch_size

        print('Create Sampler...')

    def getbatch(self):
        random_data = random.sample(list(self.data), self.batch_size)
        z = np.random.normal(0, 1, [self.batch_size, 1024])
        features = list()
        t_captions = list()
        f_captions = list()
        for img in random_data:
            features.append(self.data[img]['feature'])
            cur_t_caption = self.data[img]['caption']
            # -1 for not using <END>
            sample_t_caption = cur_t_caption[np.random.choice(cur_t_caption.shape[0], 1, replace=False), :-1]
            t_captions.append(sample_t_caption)
            while True:
                f_img = random.sample(list(self.data), 1)
                if f_img != img:
                    cur_f_caption = self.data[img]['caption']
                    break
            sample_f_caption = cur_f_caption[np.random.choice(cur_f_caption.shape[0], 1, replace=False), :-1]
            f_captions.append(sample_f_caption)

        features = np.vstack(features)
        t_captions = np.vstack(t_captions)
        f_captions = np.vstack(f_captions)

        return z, features, t_captions, f_captions


class Pretrain_Loader:
    def __init__(self, data, sess):
        self.data = data
        self.sess = sess
        self.features = list()
        self.captions = list()
        for key in tqdm(self.data.keys()):
            for _ in range(self.data[key]['caption'].shape[0]):
                self.features.append(self.data[key]['feature'])
            self.captions.append(self.data[key]['caption'])
        self.features = np.vstack(self.features)
        self.captions = np.vstack(self.captions)

        n_example = self.captions.shape[0]
        index = np.arange(n_example)
        np.random.shuffle(index)
        # do random shuffle
        self.features = self.features[index]
        self.captions = self.captions[index]

        # Use placeholder to prevent too large numpy array
        self.features_tf = tf.placeholder(dtype=self.features.dtype, shape=self.features.shape)
        self.captions_tf = tf.placeholder(dtype=self.captions.dtype, shape=self.captions.shape)
        self.created_data = None

    def create(self, num_epochs=1, batch_size=128, shuffle=False):
        created_data = collections.namedtuple('Data', 'iterator, captions, features')

        dataset = tf.data.Dataset.from_tensor_slices((self.captions_tf, self.features_tf))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_captions, next_features = iterator.get_next()

        self.created_data = created_data(
            iterator=iterator,
            captions=next_captions,
            features=next_features
        )

    def initialize(self):
        if self.created_data is None:
            raise ValueError('Need create dataset first')
        self.sess.run(self.created_data.iterator.initializer, feed_dict={self.features_tf: self.features,
                                                                         self.captions_tf: self.captions})

    @property
    def batch_captions(self):
        return self.created_data.captions

    @property
    def batch_features(self):
        return self.created_data.features

def main():
    train_data = utils.load_pickle('./data/data/train_dict.pkl')

    train_data_loader = Pretrain_Loader(train_data)


if __name__ == '__main__':
    main()
