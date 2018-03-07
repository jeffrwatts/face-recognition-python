import tensorflow as tf

from datasets import mnist
from datasets import dataset_utils
from models import lenet

from datasets import mnist
from models import lenet

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_steps', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS

def preprocess_image(image, output_height, output_width, is_training):
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)
    return image

def load_batch(dataset, batch_size=32, height=28, width=28, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image, label = data_provider.get(['image', 'label'])

    image = preprocess_image(image, height, width, is_training)

    images, labels = tf.train.batch([image, label], batch_size=batch_size, allow_smaller_final_batch=True)

    return images, labels

def main(args):
    # load the dataset
    dataset = mnist.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    images, labels = load_batch(dataset, FLAGS.batch_size, is_training=True)

   # run the image through the model
    logits, endpoints = lenet.lenet(images, num_classes=10, is_training=True, dropout_keep_prob=0.5)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    # use RMSProp to optimize
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    # create train op
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

    # run training
    slim.learning.train(train_op, FLAGS.log_dir, save_summaries_secs=20, number_of_steps=FLAGS.num_steps)


if __name__ == '__main__':
    tf.app.run()
