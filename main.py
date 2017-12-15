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
flags.DEFINE_boolean('pretrained', None, 'if use pretrain model')
flags.DEFINE_boolean('checkpoint', None, 'checkpoint to restore, is None use latest')
flags.DEFINE_string('predict_image', None, 'image to be captioned')

FLAGS = flags.FLAGS


def main(unused_argv):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    vocab = utils.load_pickle(FLAGS.vocab)

    sess = tf.Session(config=config)

    caption_model = model.RNNAgent(sess, vocab, FLAGS)

    if FLAGS.mode == 'train':
        coco_data_train = utils.load_coco(FLAGS.data_dir, 'train')
        coco_data_val = utils.load_coco(FLAGS.data_dir, 'val')

        print('Successfully load data...')

        caption_model.learn(coco_data_train.captions, coco_data_train.features,
                            coco_data_val.captions, coco_data_val.features)

    elif FLAGS.mode == 'inference':
        assert FLAGS.predict_image is not None

        caption_model.inference(FLAGS.predict_image)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
