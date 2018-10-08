import tensorflow as tf


class MultiHMLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cells, reuse):
        super(MultiHMLSTMCell, self).__init__(_reuse=reuse)
        self._cells = cells

    def zero_state(self, batch_size, dtype):
        return [cell.zero_state(batch_size, dtype) for cell in self._cells]

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def call(self, input, states):
        '''
        input + states  ->  output + new_states
        input: input(t), ha_l(t-1) for l in L                       [B, I + sum(ha_l)], L == num_layers
        states: c_l(t-1), h_l(t-1), z_l(t-1) for l in num_layers     [(c=[B, h_l], h=[B, h_l], z=[B, 1]) for l in L]
        output: h_l(t) for l in num_layers                          [[B, h_l] for l in L]
        new_states: c_l(t), h_l(t), z_l(t) for l in num_layers       [(c=[B, h_l], h=[B, h_l], z=[B, 1]) for l in L]
        '''

        total_hidden_size = sum(c._h_above_size for c in self._cells)
        raw_inp = input[:, :-total_hidden_size]                     # [B, I]
        raw_h_aboves = input[:, -total_hidden_size:]                # [B, sum(s_l+1)], s_l == state_size_of_l_layer

        ha_splits = [c._h_above_size for c in self._cells]
        h_aboves = tf.split(value=raw_h_aboves, num_or_size_splits=ha_splits, axis=1)

        z_below = tf.ones([tf.shape(input)[0], 1])                  # [B, 1]
        raw_inp = tf.concat([raw_inp, z_below], axis=1)             # [B, I + 1]

        new_states = [0] * len(self._cells)
        for i, cell in enumerate(self._cells):
            with tf.variable_scope('cell_{}'.format(i)):
                cur_state = states[i]                               # (c=[B, s_i], h=[B, s_i], z=[B, 1])
                cur_inp = tf.concat([raw_inp, h_aboves[i]], axis=1, name='input_to_cell')
                                                                    # [B, s_i-1 + 1 + s_i+1]
                raw_inp, new_state = cell(cur_inp, cur_state)
                # [B, s_i-1 + s_i+1], (c=[B, s_i], h=[B, s_i], z=[B, 1])  ->  [B, s_i + 1], (c=[B, s_i], h=[B, s_i], z=[B, 1])
                new_states[i] = new_state
        output = [ns.h for ns in new_states]

        return output, new_states