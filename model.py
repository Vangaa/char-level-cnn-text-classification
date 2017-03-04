import os

import tensorflow as tf

from tf_ops import *

class SentimentPredictior(object):

    def __init__(self, sentence_dim, word_dim, letters_count,
                 filters=64, lr=0.001, dropout=0.2,
                 words_take=[2, 3, 4],
                 session=None, checkpoint_file=None):
        self.word_dim = word_dim
        self.sentence_dim = sentence_dim
        self.letters_count = letters_count
        self.filters = filters
        self.dropout = 1 - dropout
        self.words_take = words_take
        self.lr = lr

        tf.reset_default_graph()
        self.session = session or tf.Session()

        self.build()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('logs/train', self.session.graph)
        self.test_writer = tf.summary.FileWriter('logs/test')

        init = tf.global_variables_initializer()
        self.session.run(init, {self.is_train_stage: 1})

        if checkpoint_file:
            saver = tf.train.Saver()
            saver.restore(self.session, checkpoint_file)

    def fit(self, X_train, y_train,
            num_epoch=10,
            batch_size=64, eval_every=100,
            val_data=None, checkpoint_every=100,
            checkpoint_dir="models"):

        saver = tf.train.Saver()
        save_file_format = "model_loss-{loss:.4f}_epoch-{epoch}.ckpt"

        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if val_data:
            test_X, test_y = val_data[0], val_data[1]

        counter_tr = 0
        last_loss = 1e10
        for epoch in range(1, num_epoch + 1):
            for i in range(0, X_train.shape[0], batch_size):
                train_x = X_train[i:i + batch_size]
                train_y = y_train[i:i + batch_size]
                train_loss, summary = self.fit_batch(train_x, train_y,
                                                     get_summary=True)
                self.train_writer.add_summary(summary, counter_tr)
                counter_tr += 1

                if val_data and (counter_tr % eval_every) == 0:
                    loss, summary = self.evaluate(
                        test_x, test_y, get_summary=True)
                    self.test_writer.add_summary(summary, counter_tr)

                    print("{} epoch, test loss = {1:.4f}".format(epoch, loss))
                    last_loss = loss

                if counter_tr % checkpoint_every == 0:
                    save_file = os.path.join(checkpoint_dir,save_file_format.format(loss=last_loss, epoch=epoch))
                    path = saver.save(
                        self.session,
                        save_file,
                        global_step=counter_tr
                    )
                    print("Model saved into {}\n".format(path))

    def fit_batch(self, X, y, get_summary=False):
        feed_dict = {
            self.input_sentence: X,
            self.output_class: y,
            self.is_train_stage: 1
        }

        if get_summary:
            train_loss, summary, _ = self.session.run(
                [self.loss, self.merged, self.to_optimize],
                feed_dict=feed_dict
            )
            return train_loss, summary
        else:
            train_loss, _ = self.session.run(
                [self.loss, self.to_optimize],
                feed_dict=feed_dict
            )
            return train_loss

    def predict(self, X, batch_size=128):
        predictions = []
        for i in range(0, X.shape[0], batch_size):
            feed_dict = {
                self.input_sentence: X[i:i + batch_size],
                self.is_train_stage: 0
            }
            result = self.session.run(
                tf.nn.sigmoid(self.output), feed_dict=feed_dict
            )
            try:
                predictions += list(result)
            except:
                predictions += [result]

        return np.asarray(predictions)

    def evaluate(self, X_batch, y_batch, get_summary=False):
        feed_dict = {
            self.input_sentence: X_batch,
            self.output_class: y_batch,
            self.is_train_stage: 0
        }

        if get_summary:
            test_loss, summary = self.session.run(
                [self.loss, self.merged], feed_dict=feed_dict
            )
            return test_loss, summary
        else:
            test_loss = self.session.run(
                [self.loss], feed_dict=feed_dict
            )
            return test_loss

    def build(self):
        with tf.variable_scope("sentiment"):
            with tf.name_scope("inputs"):
                self.is_train_stage = tf.placeholder(tf.bool, name='is_train')
                self.input_sentence = tf.placeholder(
                    tf.int32,
                    [None, self.sentence_dim, self.word_dim],
                    name='sentence'
                )
                self.output_class = tf.placeholder(
                    tf.float32,
                    [None],
                    name='sentiment'
                )

            batch_size = tf.shape(self.input_sentence)[0]

            reduced = self.conv_reduction(self.input_sentence)
            reduced = tf.cond(self.is_train_stage,
                              lambda: tf.nn.dropout(reduced, self.dropout),
                              lambda: reduced)

            output = linear(reduced, 1, scope="out_layer")
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=output,
                labels=tf.expand_dims(self.output_class, 1)
            )

            loss = tf.reduce_mean(loss)
            tf.summary.scalar('cross_entropy', loss)

            to_optimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

            self.loss = loss
            self.to_optimize = to_optimize
            self.output = output
            self.reduced = reduced

            tf.add_to_collection('input', self.input_sentence)
            tf.add_to_collection('is_train_stage', self.is_train_stage)
            tf.add_to_collection('prediction', tf.nn.sigmoid(output))
            tf.add_to_collection('output', self.output_class)
            tf.add_to_collection('loss', loss)

    def conv_reduction(self, sentence, reuse=False):
        with tf.variable_scope("sentence_reduce", reuse=reuse):
            batch_size = tf.shape(sentence)[0]

            sentence = tf.reshape(
                sentence,
                [batch_size, self.sentence_dim * self.word_dim]
            )

            embedded = tf.one_hot(sentence, self.letters_count,
                                  on_value=1., off_value=0.,
                                  name="chars_one_hot")
            embedded = tf.reshape(
                embedded,
                [batch_size, self.sentence_dim, self.word_dim, self.letters_count]
            )

            conv = embedded
            with tf.name_scope("char_level_reduce"):
                for i, filters_koef in enumerate([(3, 1), (3, 2)]):
                    kernel, filters_coef = filters_koef
                    conv = conv2d(conv,
                                  filters=self.filters * filters_coef,
                                  kernel=[1, kernel],
                                  padding="VALID",
                                  scope="conv_chars_{}".format(i))
                    conv = batch_norm(conv,
                                      momentum=0.99,
                                      phase_train=self.is_train_stage,
                                      scope="bn_chars_{}".format(i))
                    conv = prelu(conv, "fn_chars_{}".format(i))
                    conv = tf.cond(self.is_train_stage,
                                   lambda: tf.nn.dropout(conv, self.dropout),
                                   lambda: conv)

                new_shape = 2*self.filters*(self.word_dim-6+2)
                flatten = tf.reshape(conv,
                                     [batch_size, self.sentence_dim, new_shape, 1])

            with tf.name_scope("word_level_reduce"):
                arr = []
                for words in self.words_take:
                    conv = conv2d(flatten,
                                  filters=256,
                                  kernel=[words, new_shape],
                                  padding="VALID",
                                  scope="conv_words_{}".format(words))
                    conv = batch_norm(conv,
                                      momentum=0.99,
                                      phase_train=self.is_train_stage,
                                      scope="bn_words_{}".format(words))
                    conv = prelu(conv, "fn_words_{}".format(words))
                    arr.append(tf.reduce_max(conv, axis=1))

                concated = tf.concat(arr, 2)
                flat = tf.reshape(concated, [batch_size, 256 * 3])

            return flat
