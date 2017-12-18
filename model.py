import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import os
import utils
from data_loader import Sampler, Pretrain_Loader


class RNNAgent:
    def __init__(self, sess, vocab, params):

        # Build the network
        self.sess = sess
        self.params = params
        self.global_step = tf.train.get_or_create_global_step()
        self.rnn = RNN(sess, self.global_step, vocab, params)
        self.saver = tf.train.Saver()

        if not self.params.model_dir:
            raise ValueError('Need to provide model directory')

        if not os.path.exists(self.params.model_dir):
            os.makedirs(self.params.model_dir)

        self.summary_dir = os.path.join(self.params.model_dir, 'log')
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

    def learn(self, train_dict_data, val_dict_data):
        # First step: Prepare data for training

        train_data_loader = Pretrain_Loader(train_dict_data, self.sess)
        print('Create pre-trained training data loader...')
        val_data_loader = Pretrain_Loader(val_dict_data, self.sess)
        print('Create pre-trained validation data loader...')

        # prepare summary for tensorboard
        g_train_loss = tf.placeholder(dtype=tf.float32, shape=())
        g_train_summary = tf.summary.scalar('Train_avg_loss', g_train_loss)
        g_val_loss = tf.placeholder(dtype=tf.float32, shape=())
        g_val_summary = tf.summary.scalar('Val_avg_loss', g_val_loss)

        train_summary_op = tf.summary.merge([g_train_summary])
        val_summary_op = tf.summary.merge([g_val_summary])

        # Set up data

        train_data_loader.create(self.params.epochs_per_eval, self.params.batch_size, shuffle=True)
        val_data_loader.create(batch_size=self.params.batch_size, shuffle=False)

        self.sess.run(tf.global_variables_initializer())

        # if use pre-trained model, reload checkpoints
        if self.params.pretrained:
            checkpoint = tf.train.latest_checkpoint(self.params.model_dir) \
                if not self.params.checkpoint else self.params.checkpoint

            if checkpoint:
                print("Loading model checkpoint {}...\n".format(checkpoint))
                self.saver.restore(self.sess, checkpoint)
            else:
                raise FileNotFoundError('Can not find the checkpoint')

        for _ in range(self.params.num_epochs // self.params.epochs_per_eval):
            train_data_loader.initialize()
            val_data_loader.initialize()
            while True:
                total_train_loss = []
                try:
                    train_obs_batch, train_acs_batch = self.sess.run([train_data_loader.batch_captions,
                                                                      train_data_loader.batch_features])
                    train_loss, step = self.rnn.train(train_obs_batch, train_acs_batch)
                    total_train_loss.append(train_loss)
                    if step % self.params.summary_freq == 0:
                        print("step: {} Train loss= {:.4f}".format(step, train_loss))
                        train_summary = self.sess.run(train_summary_op,
                                                      feed_dict={g_train_loss: np.mean(total_train_loss)})
                        self.summary_writer.add_summary(train_summary, global_step=step)
                    if step % self.params.save_freq == 0:
                        self.saver.save(self.sess, os.path.join(self.params.model_dir, 'model'),
                                        global_step=self.global_step)

                except tf.errors.OutOfRangeError:
                    total_val_loss = []
                    while True:
                        try:
                            val_obs_batch, val_acs_batch = self.sess.run([val_data_loader.batch_captions,
                                                                          val_data_loader.batch_features])
                            val_loss = self.rnn.eval(val_obs_batch, val_acs_batch)
                            total_val_loss.append(val_loss)
                        except tf.errors.OutOfRangeError:
                            print('step: {} Validation loss= {:.4f}'.format(step, np.mean(total_val_loss)))
                            val_summary = self.sess.run(val_summary_op,
                                                        feed_dict={g_val_loss: np.mean(total_val_loss)})
                            self.summary_writer.add_summary(val_summary, global_step=step)
                            break
                    break

        print('Optimization done')

    def inference(self, image):

        self.sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.latest_checkpoint(self.params.model_dir) \
            if not self.params.checkpoint else self.params.checkpoint

        if not checkpoint:
            raise FileNotFoundError('Checkpoint in not found in {}'.format(self.params.model_dir))
        else:
            print("Loading model checkpoint {}...".format(checkpoint))
            self.saver.restore(self.sess, checkpoint)
        feature_extractor = utils.feature_extractor('VGG19')
        features = utils.feature_extraction(feature_extractor, image)

        predicted_sequence = self.rnn.predict(features)

        print(predicted_sequence)


class RNN:
    def __init__(self, sess, global_step, vocab, params, max_time=16, hidden_dim=512, embed_dim=512):
        self.sess = sess

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

        self.params = params
        # Word embedding weights initializer
        self.initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        with tf.variable_scope('generator'):
            self.captions = tf.placeholder(dtype=tf.int32,
                                           shape=[None, max_time+1])
            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=[None, 4096])
            self.caption_in, self.caption_out = self._process_caption(self.captions)
            self.global_step = global_step

            self.logits, self.loss, self.train_op, self.predict_sequence = self._build_rnn()

    @staticmethod
    def _process_caption(captions):
        caption_in = captions[:, :-1]
        caption_out = captions[:, 1:]

        return caption_in, caption_out

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

    def _model(self, features, captions=None, sample=True):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            image_embeddings = self._feature_embedding(features)
            batch_size = tf.shape(features)[0]
            lstm_cell = rnn.BasicLSTMCell(self.L)
            zero_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            _, initial_state = lstm_cell(image_embeddings, zero_state)

            if not sample:
                assert captions is not None
                wordvec = self._word_embedding(captions)
                outputs, _ = tf.nn.dynamic_rnn(lstm_cell, wordvec, initial_state=initial_state)
                outputs = self._logits_embedding(outputs)  # unnormalized log prob
            else:
                word_list = list()
                sample_word = self.start
                state = initial_state
                for _ in range(self.T):
                    wordvec = self._word_embedding(sample_word)
                    wordvec.set_shape([self.E])
                    wordvec = tf.expand_dims(wordvec, 0)
                    output, state = lstm_cell(wordvec, state)

                    logits = self._logits_embedding(output)
                    sample_word = tf.squeeze(tf.multinomial(logits, 1))
                    word_list.append(sample_word)
                    outputs = tf.stack(word_list)

        return outputs

    def _build_rnn(self):
        # Output
        train_logits = self._model(self.features, self.caption_in, sample=False)

        # loss
        loss = self._loss_function(logits=train_logits, labels=self.caption_out)
        # Optimizer

        update_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        '''
        def _decay_fn(decay_learning_rate, decay_global_step):
            return tf.train.exponential_decay(learning_rate=decay_learning_rate,
                                              global_step=decay_global_step,
                                              decay_steps=self.params.decay_step,
                                              decay_rate=self.params.decay_rate)
        '''

        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=self.global_step,
                                      var_list=update_params)

        predict_sequence = self._model(self.features, sample=True)

        return train_logits, loss, train_op, predict_sequence

    def predict(self, features):

        output = self.sess.run(self.predict_sequence, feed_dict={self.features: features})

        sequences = list()
        for idx in output:
            if idx == self.end:
                break
            else:
                sequences.append(self.idx2word[idx])

        sequences = ' '.join(sequences)

        return sequences

    def eval(self, captions, features):

        return self.sess.run(self.loss, feed_dict={self.captions: captions,
                                                   self.features: features})

    def train(self, captions, features):

        run_op = [self.loss, self.train_op, self.global_step]
        loss, _, step = self.sess.run(run_op, feed_dict={self.captions: captions,
                                                         self.features: features})
        return loss, step


class Discriminator:
    def __init__(self, sess, global_step, vocab, max_time=16, hidden_dim=512, embed_dim=512):
        self.sess = sess

        # vocabulary: word to index dict
        self.vocab = vocab

        # inverse vocabulary: index to word dict
        self.idx2word = {i: w for w, i in vocab.items()}

        self.start = vocab['<START>']

        self.T = max_time

        self.L = hidden_dim

        self.E = embed_dim

        self.V = len(vocab)

        self.initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        with tf.variable_scope('discriminator'):
            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=[None, 4096])
            # captions in real
            self.captions_R = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None])
            # captions from G
            self.captions_G = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None])
            # captions from different image
            self.captions_F = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None])

            self.global_step = global_step

            self.loss, self.train_op = self._build_discriminator()

    def _word_embedding(self, inputs):
        # TODO: issue if wordvec weights of g and d are the same
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

    def _model(self, features, captions):
        """

        :param captions: [batch_size, time_step]
        :param features: [batch_size, 4096]
        :return: final LSTM output
        """
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            tf.concat([tf.expand_dims(self.start,0)[None, :], captions], axis=1)
            lstm_cell = rnn.BasicLSTMCell(self.L)

            wordvec = self._word_embedding(captions)
            lstm_output, _ = tf.nn.dynamic_rnn(lstm_cell, wordvec, dtype=tf.float32)
            lstm_output = lstm_output[:, -1, :] # get final output
            image_embeddings = self._feature_embedding(features)

            output = tf.nn.sigmoid(tf.tensordot(lstm_output, image_embeddings, axes=[[1], [1]]))

        return output

    def _build_discriminator(self):
        s_r = self._model(self.features, self.captions_R)
        s_g = self._model(self.features, self.captions_G)
        s_f = self._model(self.features, self.captions_F)

        loss_r = tf.log(s_r + 1e-12)
        loss_g = tf.log((1 - s_g) + 1e-12)
        loss_f = tf.log((1-s_f) + 1e-12)

        d_loss = -(loss_r + loss_f + loss_g)  # negative because of maximization
        optimizer = tf.train.AdamOptimizer(1e-4)
        update_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        train_op = optimizer.minimize(d_loss, var_list=update_params)

        return d_loss, train_op

    def train(self, features, captions_r, captions_g, captions_f):

        run_op = [self.loss, self.train_op]
        loss, _ = self.sess.run(run_op, feed_dict={self.captions_R: captions_r,
                                                   self.captions_F: captions_f,
                                                   self.captions_G: captions_g,
                                                   self.features: features})
        return loss

    def eval(self, features, captions_r, captions_g, captions_f):

        return self.sess.run(self.loss, feed_dict={self.captions_R: captions_r,
                                                   self.captions_F: captions_f,
                                                   self.captions_G: captions_g,
                                                   self.features: features})
