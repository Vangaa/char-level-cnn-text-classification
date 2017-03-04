from sklearn.cross_validation import train_test_split
import tensorflow as tf

from model import SentimentPredictior
from utils import prepare_train_data, MAX_SENT_LEN, MAX_WORD_LEN, LETTERS_COUNT



tf.flags.DEFINE_float("test_perc", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("pos_data_file", "./data/pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("neg_data_file", "./data/neg.txt", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_string("words_take", "2,3,4", "Comma-separated filter sizes (default: '2,3,4')")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate (default: 1e-4)")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout", 0.2, "Dropout keep probability (default: 0.2)")

# Training parameters
tf.flags.DEFINE_string("checkpoint_dir", "models", "Checkpoints dir (default: models)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr, value))
print("")

print("Dataset preparing...")
train_X, train_y = prepare_train_data(FLAGS.pos_data_file, FLAGS.neg_data_file)
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y,
                                                    test_size=FLAGS.test_perc)

model = SentimentPredictior(
    MAX_SENT_LEN, MAX_WORD_LEN, LETTERS_COUNT,
    filters=FLAGS.num_filters,
    lr=FLAGS.learning_rate,
    words_take=[int(i) for i in FLAGS.words_take.split(",")],
    dropout=FLAGS.dropout
)

print("Start training...")
model.fit(train_X, train_y,
    num_epoch=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
    eval_every=FLAGS.evaluate_every, val_data=(test_X, test_y),
    checkpoint_every=FLAGS.checkpoint_every,
    checkpoint_dir=FLAGS.checkpoint_dir)
