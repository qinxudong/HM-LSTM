from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, viz_char_boundaries, plot_losses, Generator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_name1 = 'tinyshakespeare.txt'
file_name2 = 'variousscripts.txt'
file_name3 = 'text8.txt'
file_name3_index = 'text8_index.csv'

# with open(file_name3, 'r') as f:
#     raw_data = f.read()
#     print('Data length: ', len(raw_data))
#
# vocab = set(raw_data)
# vocab_size = len(vocab)
# idx_to_vocab = dict(enumerate(vocab))
# vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
# data = [vocab_to_idx[c] for c in raw_data]
# del raw_data
# data_df = pd.DataFrame(data)
# data_df.to_csv(file_name3_index)

raw_data = pd.read_csv(file_name3_index, index_col=0)
vocab_size = raw_data.max()[0] + 1
data_all = np.squeeze(raw_data.values)
data = data_all[: len(data_all) // 10]
print('Data length: ', len(data))

'''
    记录：
    直接读取100m字符串数据时，python程序占内存13%；
    读取366M的转为列表后保存的字符串数据，在调用eval()时python程序直接飙到90%以上然后SIGKILL；
    读取1.1G的csv数据并处理，python程序占内存最高到27%。
'''



num_steps = 500
batch_size = 8
num_classes = vocab_size
learning_rate = 1e-3
num_layers = 3
num_epochs = 3

task = 'classification'
network = HMLSTMNetwork(output_size=vocab_size, input_size=vocab_size, num_layers=num_layers, embed_size=1024,
                        out_hidden_size=512, hidden_state_sizes=512, task=task)
generator = Generator(data, batch_size, num_steps, num_epochs)
losses = network.train_on_generator(generator, variable_path='weights/{}_{}epochs'.format(task, num_epochs),
                                    load_weights=False)
plot_losses(losses)
plt.savefig('figs/losses_of_{}_{}layers{}epochs'.format(task, num_layers, num_epochs))
# plt.show()

# predictions = network.predict(batches_in[-1], variable_path='weights/{}_{}epochs'.format(task, num_epochs))
# boundaries = network.predict_boundaries(batches_in[-1], variable_path='weights/{}_{}epochs'.format(task, num_epochs))
#
# viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])





