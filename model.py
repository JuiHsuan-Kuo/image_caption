import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import collections
import numpy as np
import os
import utils


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

    def _data_loader(self, captions, features, num_epochs=1, batch_size=128, shuffle=False):
        data = collections.namedtuple('Data', 'iterator, captions, features')

        dataset = tf.data.Dataset.from_tensor_slices((captions, features))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_captions, next_features = iterator.get_next()

        return data(
            iterator=iterator,
            captions=next_captions,
            features=next_features
        )

    def learn(self, train_captions, train_features, val_captions, val_features):
        # First step: Prepare data for training

        # prepare data to feed
        train_captions_tf = tf.placeholder(dtype=train_captions.dtype,
                                           shape=[None, train_captions.shape[1]])
        train_features_tf = tf.placeholder(dtype=train_features.dtype,
                                           shape=[None, train_features.shape[1]])
        val_captions_tf = tf.placeholder(dtype=val_captions.dtype,
                                         shape=[None, val_captions.shape[1]])
        val_features_tf = tf.placeholder(dtype=val_features.dtype,
                                         shape=[None, val_features.shape[1]])

        # prepare summary for tensorboard
        g_train_loss = tf.placeholder(dtype=tf.float32, shape=())
        g_train_summary = tf.summary.scalar('Train_avg_loss', g_train_loss)
        g_val_loss = tf.placeholder(dtype=tf.float32, shape=())
        g_val_summary = tf.summary.scalar('Val_avg_loss', g_val_loss)

        train_summary_op = tf.summary.merge([g_train_summary])
        val_summary_op = tf.summary.merge([g_val_summary])

        # Set up data

        train_data = self._data_loader(train_captions_tf, train_features_tf, self.params.epochs_per_eval,
                                       self.params.batch_size, shuffle=True)
        val_data = self._data_loader(val_captions_tf, val_features_tf, batch_size=self.params.batch_size)

        self.sess.run(tf.global_variables_initializer())

        # if pretrain, reload checkpoints
        if self.params.pretrained:
            checkpoint = tf.train.latest_checkpoint(self.params.model_dir) \
                if not self.params.checkpoint else self.params.checkpoint

            if checkpoint:
                print("Loading model checkpoint {}...\n".format(checkpoint))
                self.saver.restore(self.sess, checkpoint)
            else:
                raise ValueError('Can not find the checkpoint')

        for _ in range(self.params.num_epochs // self.params.epochs_per_eval):
            self.sess.run(train_data.iterator.initializer, feed_dict={train_captions_tf: train_captions,
                                                                      train_features_tf: train_features})
            self.sess.run(val_data.iterator.initializer, feed_dict={val_captions_tf: val_captions,
                                                                    val_features_tf: val_features})
            while True:
                total_train_loss = []
                try:
                    train_obs_batch, train_acs_batch = self.sess.run([train_data.captions, train_data.features])
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
                            val_obs_batch, val_acs_batch = self.sess.run([val_data.captions, val_data.features])
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
            raise ValueError('Checkpoint in not found in {}'.format(self.params.model_dir))
        else:
            print("Loading model checkpoint {}...".format(checkpoint))
            self.saver.restore(self.sess, checkpoint)
        feature_exctrator = utils.feature_extractor('VGG19')
        features = utils.feature_extraction(feature_exctrator, image)

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

    def _model(self, features, captions=None, is_training=True):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            image_embeddings = self._feature_embedding(features)
            batch_size = tf.shape(features)[0]
            lstm_cell = rnn.BasicLSTMCell(self.L)
            zero_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            _, initial_state = lstm_cell(image_embeddings, zero_state)

            if is_training:
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
                    sample_word = tf.squeeze(tf.argmax(logits, axis=1))
                    word_list.append(sample_word)
                    outputs = tf.stack(word_list)

        return outputs

    def _build_rnn(self):
        # Output
        train_logits = self._model(self.features, self.caption_in, is_training=True)

        # loss
        loss = self._loss_function(logits=train_logits, labels=self.caption_out)
        # Optimizer

        def _decay_fn(decay_learning_rate, decay_global_step):
            return tf.train.exponential_decay(learning_rate=decay_learning_rate,
                                              global_step=decay_global_step,
                                              decay_steps=self.params.decay_step,
                                              decay_rate=self.params.decay_rate)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=self.global_step,
            learning_rate=0.001,
            optimizer='Adam',
            learning_rate_decay_fn=_decay_fn
        )

        predict_sequence = self._model(self.features, is_training=False)

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
    def __init__(self, env, sess, global_step):
        self.env = env
        self.sess = sess
        self.acs_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]

        with tf.variable_scope('discriminator'):
            self.acs_f = tf.placeholder(dtype=env.action_space.high.dtype,
                                        shape=[None, self.acs_dim])
            self.acs_r = tf.placeholder(dtype=env.action_space.high.dtype,
                                        shape=[None, self.acs_dim])
            self.obs_f = tf.placeholder(dtype=env.observation_space.high.dtype,
                                        shape=[None, self.obs_dim])
            self.obs_r = tf.placeholder(dtype=env.observation_space.high.dtype,
                                        shape=[None, self.obs_dim])
            self.input_r = tf.concat([self.obs_r, self.acs_r], axis=1)
            self.input_f = tf.concat([self.obs_f, self.acs_f], axis=1)
            self.global_step = global_step

            self.loss, self.train_op = self._build_discriminator()

            tf.summary.scalar('d_loss', self.loss)

    def _model(self, inputs):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(inputs=inputs,
                                  units=100,
                                  activation=tf.nn.tanh)
            net = tf.layers.dense(inputs=net,
                                  units=100,
                                  activation=tf.nn.tanh)
            net = tf.layers.dense(inputs=net,
                                  units=1)
        return net

    def _build_discriminator(self):
        p_r = self._model(self.input_r)
        p_f = self._model(self.input_f)
        loss_r = tf.log(p_r + 1e-12)
        loss_f = tf.log((1 - p_f) + 1e-12)
        d_loss = -(loss_r + loss_f)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(d_loss, global_step=self.global_step)

        return d_loss, train_op

    def train(self, obs_r, obs_f, acs_r, acs_f):

        _ = self.sess.run(self.train_op, feed_dict={self.obs_r: obs_r,
                                                    self.obs_f: obs_f,
                                                    self.acs_r: acs_r,
                                                    self.acs_f: acs_f})
