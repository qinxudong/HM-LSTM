import numpy as np
import tensorflow as tf
from .hmlstm_cell_ import HMLSTMCell, HMLSTMState
from .multi_hmlstm_cell_ import MultiHMLSTMCell

class HMLSTMNetwork(object):
    def __init__(self,
                 input_size=1,
                 output_size=1,
                 num_layers=3,
                 hidden_state_sizes=50,
                 out_hidden_size=100,
                 embed_size=100,
                 task='regression'):
        """
        HMLSTMNetwork is a class representing hierarchical multiscale
        long short-term memory network.

        params:
        ---
        input_size: integer, the size of an input at one timestep
        output_size: integer, the size of an output at one timestep
        num_layers: integer, the number of layers in the hmlstm
        hidden_state_size: integer or list of integers. If it is an integer,
            it is the size of the hidden state for each layer of the hmlstm.
            If it is a list, it must have length equal to the number of layers,
            and each integer of the list is the size of the hidden state for
            the layer corresponding to its index.
        out_hidden_size: integer, the size of the two hidden layers in the
            output network.
        embed_size: integer, the size of the embedding in the output network.
        task: string, one of 'regression' and 'classification'.
        """
        self._out_hidden_size = out_hidden_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._input_size = input_size
        self._output_size = output_size
        self._session = None
        self._graph = None
        self._task = task


        if type(hidden_state_sizes) is list and len(hidden_state_sizes) != num_layers:
            raise ValueError('The number of hidden states provided must be the' +
                             ' same as the number of layers.')

        if type(hidden_state_sizes) is int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        if task == 'classification':
            self._loss_function = tf.nn.softmax_cross_entropy_with_logits
        elif task == 'regression':
            self._loss_function = lambda logits, labels: tf.square((logits - labels))

        # batch_in_shape = (None, None, self._input_size)# (T, B, input_size)
        # batch_out_shape = (None, None, self._output_size)# (T, B, input_size)
        # self.batch_in = tf.placeholder(tf.float32, shape=batch_in_shape, name='batch_in')
        # self.batch_out = tf.placeholder(tf.float32, shape=batch_out_shape, name='batch_out')
        self.batch_in = tf.placeholder(tf.int32, shape=[None, None], name='batch_in')
        self.batch_out = tf.placeholder(tf.int32, shape=[None, None], name='batch_out')
        self._optimizer = tf.train.AdamOptimizer(1e-3)
        #  初始化参数
        self._initialize_output_variables()
        self._initialize_gate_variables()
        self._initialize_embedding_variables()

    def _initialize_gate_variables(self):
        with tf.variable_scope('gates_vars'):
            for l in range(self._num_layers):
                tf.get_variable('gate_{}'.format(l), [sum(self._hidden_state_sizes), 1],
                                dtype=tf.float32)# 这里每一层的gate.shape==(self._hidden_state_sizes[l], 1)，
                # 为什么要所有层求和呢？？ 答：因为这里的gate指的是在output embedding layer中的g_t^l。

    def _initialize_embedding_variables(self):
        with tf.variable_scope('embedding_vars'):
            embed_shape = [sum(self._hidden_state_sizes), self._embed_size]
            tf.get_variable('embed_weights', embed_shape, dtype=tf.float32)

    def _initialize_output_variables(self):
        with tf.variable_scope('output_module_vars'):
            tf.get_variable('b1', [1, self._out_hidden_size], dtype=tf.float32)
            tf.get_variable('b2', [1, self._out_hidden_size], dtype=tf.float32)
            tf.get_variable('b3', [1, self._output_size], dtype=tf.float32)
            tf.get_variable('w1', [self._embed_size, self._out_hidden_size], dtype=tf.float32)
            tf.get_variable('w2', [self._out_hidden_size, self._out_hidden_size], dtype=tf.float32)
            tf.get_variable('w3', [self._out_hidden_size, self._output_size], dtype=tf.float32)

    def load_variables(self, path=None):
        if self._session is None:
            self._session = tf.Session()

            saver = tf.train.Saver()
            print('loading variables...')
            assert path is not None
            saver.restore(self._session, path)
        else:
            print('self._session is not None, skip load_variables.')

    def save_variables(self, path=None):
        saver = tf.train.Saver()
        print('saving variables...')
        assert path is not None
        saver.save(self._session, path)

    def gate_input(self, hidden_states):
        '''
        gate the incoming hidden states
        hidden_states: [B, sum(h_l)]

        gated_input: [B, sum(h_l)]
        '''
        with tf.variable_scope('gates_vars', reuse=True):
            gates = []# [[B, 1] for l in range(L)]
            for l in range(self._num_layers):
                weights = tf.get_variable('gate_{}'.format(l), dtype=tf.float32)
                gates.append(tf.sigmoid(tf.matmul(hidden_states, weights)))

            split = tf.split(value=hidden_states,
                                    num_or_size_splits=self._hidden_state_sizes,
                                    axis=1)

            gated_list = []# [[B, h_l] for l in range(L)]
            for gate, hidden_state in zip(gates, split):
                gated_list.append(tf.multiply(gate, hidden_state))
            gated_input = tf.concat(gated_list, axis=1)# [B, sum(h_l)]
        return gated_input

    def embed_input(self, gated_input):
        '''
        gated_input: [B, sum(h_l)]

        embedding: [B, E], i.e. [B, embed_size]
        '''
        with tf.variable_scope('embedding_vars', reuse=True):
            embed_weights = tf.get_variable('embed_weights', dtype=tf.float32)
            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(prod)
        return embedding

    def output_module(self, embedding, outcome):
        '''
        embedding: [B, E]
        outcome: [B, output_size]

        loss: [B, output_size] or [B, 1]
        prediction: [B, output_size]
        '''
        with tf.variable_scope('output_module_vars', reuse=True):
            b1 = tf.get_variable('b1')
            b2 = tf.get_variable('b2')
            b3 = tf.get_variable('b3')
            w1 = tf.get_variable('w1')
            w2 = tf.get_variable('w2')
            w3 = tf.get_variable('w3')

            # feed forward network
            # first layer
            l1 = tf.nn.tanh(tf.matmul(embedding, w1) + b1)

            # second layer
            l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)

            # output
            prediction = tf.add(tf.matmul(l2, w3), b3, name='prediction')

            loss_args = {'logits': prediction, 'labels': outcome}
            loss = self._loss_function(**loss_args)

            if self._task == 'classification':
                loss = tf.expand_dims(loss, -1) # 这里由于tf.nn.softmax_cross_entropy_with_logits输出loss会比logits或者
                                                # labels降一维，因此expand_dims让loss回到(B, 1)。

        return loss, prediction

    def create_multicell(self, batch_size, reuse):

        def hmlstm_cell(layer):
            if layer == 0:
                h_below_size = self._input_size
            else:
                h_below_size = self._hidden_state_sizes[layer - 1]

            if layer == self._num_layers - 1:
                # doesn't matter, all zeros, but for convenience with summing
                # so the sum of ha sizes is just sum of hidden states
                # 意思是将最上层的h_above_size设为最下层的hidden_state_size
                h_above_size = self._hidden_state_sizes[0]
            else:
                h_above_size = self._hidden_state_sizes[layer + 1]

            return HMLSTMCell(self._hidden_state_sizes[layer], batch_size, h_below_size,
                              h_above_size, reuse)

        hmlstm = MultiHMLSTMCell([hmlstm_cell(l) for l in range(self._num_layers)], reuse)
        return hmlstm

    def split_out_cell_states(self, accum):
        '''
        accum: [B, H], i.e. [B, sum(h_l + h_l + 1)]

        cell_states: a list of ([B, h_l], [B, h_l], [B, 1]), with length L
        '''
        splits = []
        for size in self._hidden_state_sizes:
            splits += [size, size, 1]
        split_states = tf.split(value=accum, num_or_size_splits=splits, axis=1)
        cell_states = []
        for l in range(self._num_layers):
            c = split_states[(l * 3)]
            h = split_states[(l * 3) + 1]
            z = split_states[(l * 3) + 2]
            cell_states.append(HMLSTMState(c=c, h=h, z=z))# (cell_state, hidden_state, boundary_detector)
        return cell_states

    def get_h_aboves(self, hidden_states, batch_size, hmlstm):
        '''
        hidden_states: [[B, h_l] for l in range(L)]

        h_aboves: [B, sum(ha_l)], ha denotes h_above
        '''
        concated_hs = tf.concat(hidden_states[1:], axis=1)
        h_above_for_last_layer = tf.zeros([batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32)
        h_aboves = tf.concat([concated_hs, h_above_for_last_layer], axis=1)
        return h_aboves

    def network(self, reuse):
        batch_size = tf.shape(self.batch_in)[0]
        hmlstm = self.create_multicell(batch_size, reuse)

        def scan_rnn(accum, elem):  # accum: 堆叠所有层的c,h,z的值组成的tensor, shape==[B, H]
                                    # elem: 一个batch, 一个时间步长的输入, shape==[B, I]
            cell_states = self.split_out_cell_states(accum)     # [B, H] -> [([B, h_l], [B, h_l], [B, 1]) for l in L]
            h_aboves = self.get_h_aboves([cs.h for cs in cell_states], batch_size, hmlstm)  # [B, sum(ha_l)]
            hmlstm_in = tf.concat((elem, h_aboves), axis=1)     # [B, I] + [B, sum(ha_l)] -> [B, I + sum(ha_l)]
            _, state = hmlstm(hmlstm_in, cell_states)
            # a list of (c=[B, h_l], h=[B, h_l], z=[B, 1])  ->  a list of (c=[B, h_l], h=[B, h_l], z=[B, 1])
            concated_states = [tf.concat(tuple(s), axis=1) for s in state]
            return tf.concat(concated_states, axis=1)           # [B, H]

        elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers
        initial = tf.zeros([batch_size, elem_len])              # [B, H], H = elem_len
        batch_in_onehot = tf.one_hot(self.batch_in, self._input_size)   # [B, T] -> [B, T, I]
        batch_in_trans = tf.transpose(batch_in_onehot, [1, 0, 2])       # [T, B, I]
        states = tf.scan(scan_rnn, batch_in_trans, initial)     # [T, B, H]
        
        def map_indicators(elem):
            state = self.split_out_cell_states(elem)
            return tf.concat([l.z for l in state], axis=1)
        raw_indicators = tf.map_fn(map_indicators, states)      # [T, B, L]
        indicators = tf.transpose(raw_indicators, [1, 2, 0])    # [B, L, T]
        batch_out_onehot = tf.one_hot(self.batch_out, self._output_size)    # [B, T] -> [B, T, O], O = output_size
        batch_out_trans = tf.transpose(batch_out_onehot, [1, 0, 2])         # [T, B, O]
        to_map = tf.concat((states, batch_out_trans), axis=2)   # [T, B, H+O]

        def map_output(elem):
            splits = tf.constant([elem_len, self._output_size])
            cell_states, outcome = tf.split(value=elem, num_or_size_splits=splits, axis=1)
            hs = [s.h for s in self.split_out_cell_states(cell_states)]
            gated = self.gate_input(tf.concat(hs, axis=1))      # [B, sum(h_l)]
            embeded = self.embed_input(gated)                   # [B, E] E = embeded_size
            loss, prediction = self.output_module(embeded, outcome)
                                                                # [B, 1], [B, output_size] for classification
                                                                # [B, output_size], [B, output_size] for regression(usually output_size == 1)
            return tf.concat((loss, prediction), axis=1)        # [B, 1+output_size] or [B, 2*output_size]

        mapped = tf.map_fn(map_output, to_map)                  # [T, B, 1+output_size] or [T, B, 2*output_size]
        loss = tf.reduce_mean(mapped[:, :, :-self._output_size])# scalar
        predictions = mapped[:, :, -self._output_size:]         # [T, B, output_size]
        train = self._optimizer.minimize(loss)

        return train, loss, indicators, predictions

    def _get_graph(self):
        if self._graph is None:
            self._graph = self.network(reuse=False)
        return self._graph

    def train(self, batches_in, batches_out, variable_path='weights/unnamed_weights', load_weights=False, epochs=1):
        """
        Train the network.

        params:
        ---
        batches_in: a 4 dimensional numpy array. The dimensions should be
            [num_batches, batch_size, num_timesteps, input_size]
            These represent the input at each time step for each batch.
        batches_out: a 4 dimensional numpy array. The dimensions should be
            [num_batches, batch_size, num_timesteps, output_size]
            These represent the output at each time step for each batch.
        variable_path: the path to which variable values will be saved and/or
            loaded
        load_weights: bool, whether to load variables prior to training
        save_weights: bool, whether to save variables after training
        epochs: integer, number of epochs
        """

        train, loss, _, _ = self._get_graph()
        losses = []

        if not load_weights:
            if self._session is None:
                self._session = tf.Session()
                init = tf.global_variables_initializer()
                self._session.run(init)
        else:
            self.load_variables(variable_path)

        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            for batch_in, batch_out in zip(batches_in, batches_out):
                fetches = [train, loss]
                feed_dict = {
                    self.batch_in: batch_in,     # (B, T)
                    self.batch_out: batch_out    # (B, T)
                }
                _, _loss = self._session.run(fetches, feed_dict)
                print('loss:', _loss)
                losses.append(_loss)

        self.save_variables(variable_path)
        return losses

    def train_on_generator(self, generator, variable_path='weights/unnamed_weights', load_weights=False):
        train, loss, _, _ = self._get_graph()
        losses = []

        if not load_weights:
            if self._session is None:
                self._session = tf.Session()
                init = tf.global_variables_initializer()
                self._session.run(init)
        else:
            self.load_variables(variable_path)

        for idx, epoch in enumerate(generator.gen_epochs()):
            print('\nEpoch {}'.format(idx))
            for batch_idx, (batch_in, batch_out) in enumerate(epoch):
                fetches = [train, loss]
                feed_dict = {
                    self.batch_in: batch_in,    # (B, T)
                    self.batch_out: batch_out   # (B, T)
                }
                _, _loss = self._session.run(fetches, feed_dict)
                print('Loss of batch {}: '.format(batch_idx), _loss)
                losses.append(_loss)

        self.save_variables(variable_path)
        return losses

    def predict(self, batch, variable_path=None, return_gradients=False):
        """
        Make predictions.

        params:
        ---
        batch: batch for which to make predictions. should have dimensions
            [batch_size, num_timesteps, output_size]
        variable_path: string. If there is no active session in the network
            object (i.e. it has not yet been used to train or predict, or the
            tensorflow session has been manually closed), variables will be
            loaded from the provided path. Otherwise variables already present
            in the session will be used.

        returns:
        ---
        predictions for the batch
        """

        batch = np.array(batch)
        _, _, _, predictions = self._get_graph()

        if self._session is None:
            try:
                self.load_variables(variable_path)
            except:
                raise RuntimeError('Session unitialized and no variables saved at path {}'.format(variable_path))

        batch_out_size = (batch.shape[1], batch.shape[0], self._output_size)
        gradients = tf.gradients(predictions[-1:, :], self.batch_in)
        feed_dict = {self.batch_in: np.swapaxes(batch, 0, 1),
                     self.batch_out: np.zeros(batch_out_size)}
        _predictions, _gradients = self._session.run([predictions, gradients], feed_dict)

        if return_gradients:
            return tuple(np.swapaxes(r, 0, 1) for r in (_predictions, _gradients[0]))
        return np.swapaxes(_predictions, 0, 1)

    def predict_boundaries(self, batch, variable_path=None):
        """
        Find indicator values for every layer at every timestep.

        params:
        ---
        batch: batch for which to make predictions. should have dimensions
            [batch_size, num_timesteps, output_size]
        variable_path: string. If there is no active session in the network
            object (i.e. it has not yet been used to train or predict, or the
            tensorflow session has been manually closed), variables will be
            loaded from the provided path. Otherwise variables already present
            in the session will be used.

        returns:
        ---
        indicator values for ever layer at every timestep
        """

        batch = np.array(batch)
        _, _, indicators, _ = self._get_graph()

        if self._session is None:
            try:
                self.load_variables(variable_path)
            except:
                raise RuntimeError('Session unitialized and no variables saved at path {}'.format(variable_path))

        feed_dict = {self.batch_in: np.swapaxes(batch, 0, 1)}
        _indicators = self._session.run(indicators, feed_dict)

        return np.array(_indicators)






















