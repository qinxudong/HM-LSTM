import numpy as np
import matplotlib.pyplot as plt
from hmlstm import HMLSTMNetwork, plot_indicators, prepare_inputs, get_text, convert_to_batches, \
    viz_char_boundaries, plot_losses
import tensorflow as tf

num_signals = 300
signal_length = 400
x = np.linspace(start=0, stop=50 * np.pi, num=signal_length)  # 形状为(400,)的array
signals = [np.random.normal(0, .5, size=signal_length)  # 形状为(400,)的array
           + (2 * np.sin(.6 * x + np.random.random() * 10))  # 形状为(400,)的array
           + (5 * np.sin(.1 * x + np.random.random() * 10))  # 形状为(400,)的array
           for _ in range(num_signals)]  # 包含300个 形状为(400,)的array 的list
# print(len(signals))
split = int(num_signals * .8)
train = signals[:split]
test = signals[split:]
# plt.figure(1)
# plt.subplot(411)
# plt.plot(x, signals[0])
# plt.subplot(412)
# plt.plot(x, signals[1])
# plt.subplot(413)
# plt.plot(x, signals[2])
# plt.subplot(414)
# plt.plot(x, signals[3])
# plt.show()
batches_in, batches_out = convert_to_batches(signals, batch_size=10)
# batches_in: 30 batches, 10 samples/batch, 前399个x/sample, 1个浮点数/x  ->  (30, 10, 399, 1)
# batches_in: 30 batches, 10 samples/batch, 后399个x/sample, 1个浮点数/x  ->  (30, 10, 399, 1)
# print(batches_in.shape, batches_out.shape)

task = 'regression'
epochs = 50
num_layers = 2
tf.reset_default_graph()
network = HMLSTMNetwork(input_size=1, output_size=1, task=task, hidden_state_sizes=30,
                        embed_size=50, out_hidden_size=30, num_layers=num_layers)
losses = network.train(batches_in[:-1], batches_out[:-1], epochs=epochs, load_weights=False,
              save_weights=True, variable_path='weights/{}_{}epochs'.format(task, epochs))
boundaries = network.predict_boundaries(batches_in[-1])     # [B, L, T]
predictions = network.predict(batches_in[-1])               # [B, T, O]

plot_indicators(truth=np.squeeze(batches_out[-1][0]), prediction=np.squeeze(predictions[0]), indicators=boundaries[0])
                # [T, O]                  # [T, O]                   # [L, T]
plt.savefig('figs/prediction_of_{}_{}layers{}epochs'.format(task, num_layers, epochs))
# plt.show()
plot_losses(losses)
plt.savefig('figs/losses_of_{}_{}layers{}epochs'.format(task, num_layers, epochs))
# plt.show()
