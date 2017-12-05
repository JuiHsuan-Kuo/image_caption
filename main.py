import tensorflow as tf

import utils
import model

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'train and validation data directory')
flags.DEFINE_string('model_dir', None, 'where to store checkpoints')
flags.DEFINE_string('vocab', None, 'vocab plk path')
flags.DEFINE_string('mode', None, 'train, validation, test')
flags.DEFINE_integer('num_epochs', 1, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epochs_per_eval', 1, 'epochs between evaluation')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('save_freq', 5000, 'steps between saving two checkpoints')
flags.DEFINE_integer('decay_step', 100000, 'steps per decay')
flags.DEFINE_integer('decay_rate', 0.1, 'decay rate')
flags.DEFINE_string('predict_image', None, 'image to be captioned')

FLAGS = flags.FLAGS


def main(unused_argv):

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_freq,
                                        keep_checkpoint_max=10,
                                        session_config=session_config)

    params = {
        'learning_rate': FLAGS.learning_rate,
        'decay_step': FLAGS.decay_step,
        'decay_rate': FLAGS.decay_rate,
    }
    vocab = utils.load_pickle(FLAGS.vocab)
    caption_model = model.Captioner(vocab)
    caption_model.build_estimator(config=run_config,
                                  model_dir=FLAGS.model_dir,
                                  params=params)

    if FLAGS.mode == 'train':
        coco_data = utils.load_coco(FLAGS.data_dir, FLAGS.mode)
        print('Successfully loading data')

        for _ in range(FLAGS.num_epochs // FLAGS.epochs_per_eval):
            caption_model.train(captions=coco_data.captions,
                                features=coco_data.features,
                                batch_size=FLAGS.batch_size,
                                epochs=FLAGS.epochs_per_eval)
    elif FLAGS.mode == 'inference':

        assert FLAGS.predict_image is not None
        caption_model.predict(FLAGS.predict_image)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
