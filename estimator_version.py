import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn


class Captioner:
    def __init__(self, vocab, max_time=16, embed_dim=512, hidden_dim=512):

        # image features: VGG19 fc2 features, shape(n_example, 4096)

        # sentences input captions: time series tokens, shape(n_example, 16) for training

        # max time_step
        self.T = max_time

        # vocabulary: word to index dict
        self.vocab = vocab

        # inverse vocabulary: index to word dict
        self.idx2word = {i: w for w, i in vocab.items()}

        # LSTM hidden state
        self.L = hidden_dim

        # word_vector and image feature embedded dim
        self.E = embed_dim

        # vocabulary size: For input and output one-hot encoding
        self.V = len(vocab)

        # int, indicating NULL number
        self.null = vocab['<NULL>']

        # int, indicating START number
        self.start = vocab['<START>']

        # int, indicating END numbber
        self.end = vocab['<END>']

        # Word embedding weights initializer
        self.initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Estimator
        self.Estimator = None

    def _process_caption(self, captions):
        caption_in = captions[:, :-1]
        caption_out = captions[:, 1:]

        return caption_in, caption_out

    def _input_fn(self, features, captions=None, num_epochs=1, batch_size=128, shuffle=False, is_predict=False):

        if not is_predict:
            if captions is None:
                raise ValueError('Captions are not provided')
            caption_in, caption_out = self._process_caption(captions)
            return tf.estimator.inputs.numpy_input_fn(
                x={'caption_in': caption_in, 'image_feature': features.astype(np.float32)},
                y=caption_out,
                batch_size=batch_size,
                num_epochs=num_epochs,
                shuffle=shuffle,
                queue_capacity=10000
            )
        else:
            return tf.estimator.inputs.numpy_input_fn(
                x={'image_feature': features.astype(np.float32)},
                batch_size=1,
                num_epochs=1,
                shuffle=False
            )

    def _word_embedding(self, inputs):
        """

        :param inputs: captions, (batch_size)
        :return: embedding wordvec, (batch_size, time_step, embedding_size)
        """
        with tf.variable_scope('word_embedding', reuse=tf.AUTO_REUSE):
            w_embed = tf.get_variable(name='embedding_weights', shape=[self.V, self.E],
                                      initializer=self.initializer)
            wordvec = tf.nn.embedding_lookup(w_embed, inputs)
        return wordvec

    def _feature_embedding(self, inputs):
        with tf.variable_scope('feature_embedding') as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=self.E,
                activation_fn=None,
                biases_initializer=None,
                scope=scope)

        return image_embeddings

    def _logits_embedding(self, inputs):
        """

        :param inputs: (batch_size, time_step, LSTM_size)
        :return: (batch_size, time_step, vocab_size)
        """
        with tf.variable_scope('logits') as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=self.V,
                activation_fn=None,
                scope=scope
            )
        return logits

    def _loss_function(self, logits, labels):
        with tf.name_scope('calculate_loss'):
            mask = tf.to_float(tf.not_equal(labels, self.null))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                  logits=logits)
            loss = tf.reduce_sum(tf.multiply(loss, mask))

        return loss

    def rnn_network(self, mode, features):

        with tf.variable_scope('build_network', reuse=tf.AUTO_REUSE):
            image_embeddings = self._feature_embedding(features['image_feature'])
            batch_size = tf.shape(features['image_feature'])[0]
            lstm_cell = rnn.BasicLSTMCell(self.L)
            zero_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            _, initial_state = lstm_cell(image_embeddings, zero_state)
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                wordvec = self._word_embedding(features['caption_in'])
                outputs, _ = tf.nn.dynamic_rnn(lstm_cell, wordvec, initial_state=initial_state)
                outputs = self._logits_embedding(outputs)  # unnormalized log prob
            elif mode == tf.estimator.ModeKeys.PREDICT:
                word_list = list()
                sample_word = self.start
                state = initial_state
                for _ in range(self.T):
                    wordvec = self._word_embedding(sample_word)
                    wordvec.set_shape([self.E])
                    wordvec = tf.expand_dims(wordvec, 0)
                    output, state = lstm_cell(wordvec, state)

                    logits = self._logits_embedding(output)
                    sample_word = tf.squeeze(tf.argmax(logits, axis=1))
                    word_list.append(sample_word)
                outputs = tf.stack(word_list)
            else:
                raise NameError('mode is not estimator standard key')

        print('build the network')

        return outputs

    def _model_fn(self, features, labels, mode, params):
        """

        :param features: dict of 'caption_in' and 'image_feature'
        :param labels: caption_out
        :param mode: Estimator mode
        :param params: dict: 'learning_rate', 'decay_step', 'decay_rate'
        :return:
        """

        if mode == tf.estimator.ModeKeys.EVAL:
            outputs = self.rnn_network(mode, features)

            loss = self._loss_function(outputs, labels)

        else:
            loss = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            outputs = self.rnn_network(mode, features)

            # Define loss function
            loss = self._loss_function(outputs, labels)

            global_step = tf.train.get_global_step()

            def _decay_fn(decay_learning_rate, decay_global_step):

                return tf.train.exponential_decay(learning_rate=decay_learning_rate,
                                                  global_step=decay_global_step,
                                                  decay_steps=params['decay_step'],
                                                  decay_rate=params['decay_rate'])

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step,
                learning_rate=params['learning_rate'],
                optimizer='Adam',
                learning_rate_decay_fn=_decay_fn
            )

        else:
            train_op = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            outputs = self.rnn_network(mode, features)
            predictions = {'word_sequence': outputs}
        else:
            predictions = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions=predictions
        )

    def build_estimator(self, config, model_dir, params):

        self.Estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=model_dir,
            config=config,
            params=params
        )

    def train(self, features, captions, batch_size, epochs):

        if self.Estimator is None:
            raise ValueError('Estimator is not built yet')

        self.Estimator.train(
            input_fn=self._input_fn(features, captions, num_epochs=epochs, batch_size=batch_size, shuffle=True)
        )

    def eval(self, features, captions, batch_size):

        if self.Estimator is None:
            raise ValueError('Estimator is not built yet')

        self.Estimator.evaluate(
            input_fn=self._input_fn(features, captions, num_epochs=1, batch_size=batch_size, shuffle=False)
        )

    def predict(self, image):

        if self.Estimator is None:
            raise ValueError('Estimator is not built yet')

        feature_model = utils.feature_extractor('VGG19')
        image_feature = utils.feature_extraction(feature_model, image)
        predict = list(self.Estimator.predict(input_fn=self._input_fn(image_feature, is_predict=True)))
        idxs = [p['word_sequence'] for p in predict]

        sequences = list()
        for idx in idxs:
            if idx == self.end:
                break
            else:
                sequences.append(self.idx2word[idx])

        sequences = ' '.join(sequences)

        print(sequences)


import tensorflow as tf

import utils
import model

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'train and validation data directory')
flags.DEFINE_string('model_dir', None, 'where to store checkpoints')
flags.DEFINE_string('vocab', None, 'vocab pkl path')
flags.DEFINE_string('mode', None, 'train, validation, test')
flags.DEFINE_integer('num_epochs', 1, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epochs_per_eval', 1, 'epochs between evaluation')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_integer('summary_freq', 200, 'frequency to write summary on tensorboard')
flags.DEFINE_integer('save_freq', 4000, 'steps between saving two checkpoints')
flags.DEFINE_integer('decay_step', 100000, 'steps per decay')
flags.DEFINE_float('decay_rate', 0.1, 'decay rate')
flags.DEFINE_boolean('pretrained', None, 'if use pretrain model, need True while inferencing')
flags.DEFINE_boolean('checkpoint', None, 'checkpoint to restore, is None use latest')
flags.DEFINE_string('predict_image', None, 'image to be captioned')

FLAGS = flags.FLAGS


def main(unused_argv):

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    vocab = utils.load_pickle(FLAGS.vocab)
    caption_model = model.Captioner(vocab)
    caption_model.build_estimator(config=run_config,
                                  model_dir=FLAGS.model_dir,
                                  params=params)

    if FLAGS.mode == 'train':
        coco_data_train = utils.load_coco(FLAGS.data_dir, 'train')
        coco_data_val = utils.load_coco(FLAGS.data_dir, 'val')
        print('Successfully loading data')

        for _ in range(FLAGS.num_epochs // FLAGS.epochs_per_eval):
            caption_model.train(captions=coco_data_train.captions,
                                features=coco_data_train.features,
                                batch_size=FLAGS.batch_size,
                                epochs=FLAGS.epochs_per_eval)
            caption_model.eval(captions=coco_data_val.captions,
                               features=coco_data_val.features,
                               batch_size=FLAGS.batch_size)
    elif FLAGS.mode == 'inference':

        assert FLAGS.predict_image is not None
        caption_model.predict(FLAGS.predict_image)

if __name__ == '__main__':
    tf.app.run()
