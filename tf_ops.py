import tensorflow as tf


def prelu(x, scope=None):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope, "prelu", [x]):
        leak = tf.get_variable(
            "leak",
            shape[1:],
            initializer=tf.constant_initializer(0.2)
        )
        pos = tf.nn.relu(x)

        neg = leak * (x - abs(x)) * 0.5
        return pos + neg


def linear(x, output_size,
           stddev=0.02, bias_start=1e-4, scope=None):
    with tf.variable_scope(scope, "fully_connected", [x]):
        matrix = tf.get_variable(
            "weights",
            [x.get_shape()[-1], output_size],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        bias = tf.get_variable(
            "bias",
            [output_size],
            initializer=tf.constant_initializer(bias_start)
        )
        return tf.matmul(x, matrix) + bias


def conv2d(x, filters,
           kernel=[2, 2],
           stride=[1, 1],
           stddev=0.01,
           padding='VALID',
           scope=None):

    with tf.variable_scope(scope, "conv2", [x]):
        w = tf.get_variable(
            'weights',
            kernel + [x.get_shape()[-1], filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            'biases',
            [filters],
            initializer=tf.constant_initializer(1e-4)
        )

        conv = tf.nn.conv2d(x, w,
                            strides=[1, stride[0], stride[1], 1],
                            padding=padding)
        conv = tf.nn.bias_add(conv, biases)

        return conv


def batch_norm(x, phase_train, momentum=0.9, scope=None):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope, "batch_norm", [x, phase_train]):
        gamma = tf.get_variable(
            "gamma",
            [shape[-1]],
            initializer=tf.random_normal_initializer(1., 0.01)
        )
        beta = tf.get_variable(
            "beta",
            [shape[-1]],
            initializer=tf.constant_initializer(0.)
        )

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=momentum)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
