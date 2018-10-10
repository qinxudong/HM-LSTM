import re
import numpy as np
from string import ascii_lowercase


def text(text_path, truncate_len, step_size, batch_size, num_chars=None):
    signals = load_text(text_path, truncate_len, step_size, batch_size, num_chars)

    hot = [(one_hot_encode(intext), one_hot_encode(outtext))
           for intext, outtext in signals]

    return hot


def load_text(text_path, truncate_len, step_size, batch_size, num_chars):
    with open(text_path, 'r') as f:
        text = f.read(num_chars)
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text).lower()

    signals = []
    start = 0
    while start + truncate_len < len(text):
        intext = text[start:start + truncate_len]
        outtext = text[start + 1:start + truncate_len + 1]
        signals.append((intext, outtext))
        start += step_size

    return signals


def one_hot_encode(text):
    out = np.zeros((len(text), 27))

    def get_index(char):
        try:
            return ascii_lowercase.index(char)
        except:
            return 26

    for i, t in enumerate(text):
        out[i, get_index(t)] = 1

    return out


def get_text(encoding):
    prediction = ''

    for char in np.squeeze(encoding):
        max_likelihood = np.where(char == np.max(char))[0][0]
        if max_likelihood < 26:
            prediction += ascii_lowercase[max_likelihood]
        elif max_likelihood == 26:
            prediction += ' '

    return prediction


def prepare_inputs(batch_size=10,
                   truncate_len=1000,
                   text_path='text8.txt',
                   step_size=None,
                   num_batches=None):

    if step_size is None:
        step_size = truncate_len // 2

    if num_batches is None:
        y = text(text_path, truncate_len, step_size, batch_size)
        num_batches = len(y) // batch_size
    elif num_batches is not None:
        if step_size > truncate_len:
            raise ValueError('Step size cannot be greater than truncate length')
        num_chars = batch_size * num_batches * truncate_len
        y = text(text_path, truncate_len, step_size, batch_size, num_chars)

    batches_in = []
    batches_out = []

    for batch_number in range(num_batches):
        start = batch_number * batch_size
        end = start + batch_size
        batches_in.append([i for i, _ in y[start:end]])
        batches_out.append([o for _, o in y[start:end]])

    return batches_in, batches_out

def convert_to_batches(signals, batch_size=10, steps_ahead=1):
    start = 0
    batches_in = []
    batches_out = []
    while start + batch_size <= len(signals):
        batch = signals[start: start + batch_size]

        batches_in.append(np.array([s[:-steps_ahead] for s in batch]).reshape(batch_size, -1, 1))
        batches_out.append(np.array([s[steps_ahead:] for s in batch]).reshape(batch_size, -1, 1))

        start += batch_size

    return np.array(batches_in), np.array(batches_out)
    
class Generator(object):
    def __init__(self, raw_data, batch_size, num_steps, num_epochs):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.raw_data = np.array(raw_data, dtype=np.int32)
        self.data_len = len(raw_data)
        self.batch_partition_len = self.data_len // batch_size
        self.num_batches_per_epoch = (self.batch_partition_len - 1) // num_steps
        print('Total {} batches per epoch.'.format(self.num_batches_per_epoch))
        print('Batch shape: [{}, {}]'.format(batch_size, num_steps))

    def gen_epochs(self):

        def ptb_iterator():
            data = np.zeros([self.batch_size, self.batch_partition_len], dtype=np.int32)
            for i in range(self.batch_size):
                data[i] = self.raw_data[self.batch_partition_len * i:self.batch_partition_len * (i + 1)]
            if self.num_batches_per_epoch == 0:
                raise ValueError()
            for i in range(self.num_batches_per_epoch):
                x = data[:, i * self.num_steps: (i + 1) * self.num_steps]
                y = data[:, i * self.num_steps + 1: (i + 1) * self.num_steps + 1]
                yield (x, y)

        for i in range(self.num_epochs):
            yield ptb_iterator()