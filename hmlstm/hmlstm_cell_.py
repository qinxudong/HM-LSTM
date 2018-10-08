#  原来的tensorflow.python.ops.rnn_cell_impl._linear现在放到tensorflow.contrib.rnn.python.ops.core_rnn_cell._linear。

import tensorflow as tf
import collections


HMLSTMState = collections.namedtuple('HMLSTMState', ['c', 'h', 'z'])


class HMLSTMCell(tf.nn.rnn_cell.RNNCell):
    # 这里RNNCell是所有RNN cell的基类，比如同在tf.nn.rnn_cell下的BasicRNNCell, LSTMCell等。
    # 编写一个RNN cell子类的关键在于继承RNNCell后写一个call方法，继承的RNNCell.__call__会调用这个call方法。
    def __init__(self, num_units, batch_size, h_below_size, h_above_size, reuse):
        super(HMLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._h_below_size = h_below_size
        self._h_above_size = h_above_size
        self._batch_size = batch_size

    @property
    def state_size(self):
        return (self._num_units, self._num_units, 1)

    @property
    def output_size(self):
        return self._num_units + 1

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units])
        h = tf.zeros([batch_size, self._num_units])
        z = tf.zeros([batch_size])
        return HMLSTMState(c=c, h=h, z=z)

    def call(self, input, state):
        '''
        input + state  ->  output + new_state
        input: hb(t), zb(t), ha(t-1)            [B, hb_l + 1 + ha_l]
        state: c(t-1), h(t-1), z(t-1)           (c=[B, h_l], h=[B, h_l], z=[B, 1])
        output: h(t), z(t)                      [B, h_l + 1]
        new_state: c(t), h(t), z(t)             (c=[B, h_l], h=[B, h_l], z=[B, 1])
        '''
        c = state.c                                     # [B, s_l], s_l == state_size_of_l_layer
        h = state.h                                     # [B, s_l]
        z = state.z                                     # [B, 1]
        input_splits = tf.constant([self._h_below_size, 1, self._h_above_size])
        hb, zb, ha = tf.split(value=input, num_or_size_splits=input_splits, axis=1, name='split')
                                                        # [B, s_l-1], [B, 1], [B, s_l+1]
        s_recurrent = h                                 # [B, s_l]
        s_above = tf.multiply(z, ha)                    # [B, s_l+1]
        s_below = tf.multiply(zb, hb)                   # [B, s_l-1]
        length = 4 * self._num_units + 1
        states = tf.concat([s_recurrent, s_above, s_below], axis=1)
        concat = tf.layers.Dense(units=length)(states)  # [B, 4 * s_l + 1]
        gate_splits = tf.constant([self._num_units] * 4 + [1], dtype=tf.int32)
        i, g, f, o, z_tilde = tf.split(value=concat, num_or_size_splits=gate_splits, axis=1)
        i = tf.sigmoid(i)                               # [B, s_l]
        g = tf.tanh(g)                                  # [B, s_l]
        f = tf.sigmoid(f)                               # [B, s_l]
        o = tf.sigmoid(o)                               # [B, s_l]
        new_c, new_h = self.calculate_new_state(i, g, f, o, c, h, z, zb)
        new_z = tf.expand_dims(self.calculate_new_indicator(z_tilde), -1)

        output = tf.concat((new_h, new_z), axis=1)      # [B, s_l + 1]
        new_state = HMLSTMState(c=new_c, h=new_h, z=new_z)

        return output, new_state

    def calculate_new_state(self, i, g, f, o, c, h, z, zb):
        z = tf.squeeze(z, axis=[1])
        zb = tf.squeeze(zb, axis=[1])
        new_c = tf.where(
            tf.equal(z, tf.constant(1., dtype=tf.float32)),
            tf.multiply(i, g, name='c'),
            tf.where(
                tf.equal(zb, tf.constant(0., dtype=tf.float32)),
                tf.identity(c),
                tf.add(tf.multiply(f, c), tf.multiply(i, g))
            )
        )

        new_h = tf.where(
            tf.logical_and(tf.equal(z, tf.constant(0., dtype=tf.float32)),
                                        tf.equal(zb, tf.constant(0., dtype=tf.float32))),
            tf.identity(h),
            tf.multiply(o, tf.tanh(new_c))
        )

        return new_c, new_h

    def calculate_new_indicator(self, z_tilde):
        slope_multiplier = 1
        sigmoided = tf.sigmoid(z_tilde * slope_multiplier)
        graph = tf.get_default_graph()
        with tf.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(sigmoided, name=name)

        return tf.squeeze(new_z, axis=1)
