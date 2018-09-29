import matplotlib.pyplot as plt


def plot_indicators(truth, prediction, indicators):
    plt.figure(figsize=(16, 9))
    plt.plot(truth, label='truth')
    plt.plot(prediction, label='prediction')
    plt.legend()

    colors = ['r', 'b', 'g', 'o', 'm', 'l', 'c']
    for l, layer in enumerate(indicators):
        for i, indicator in enumerate(layer):
            if indicator == 1.:
                p = 1.0 / indicators.shape[0]
                ymin = p * l
                ymax = p * (l + 1)
                plt.axvline(i, color=colors[l], ymin=ymin, ymax=ymax, alpha=.3)


def plot_losses(losses):
    plt.figure(figsize=(16, 9))
    plt.xlabel('batches')
    plt.ylabel('losses')
    plt.plot(losses)


def viz_char_boundaries(truth, predictions, indicators, row_len=60):
    start = 0
    end = row_len
    while start < len(truth):
        for l in reversed(indicators):
            print(''.join([str(int(b)) for b in l])[start:end])
        print(predictions[start:end])
        print(truth[start:end])
        print()

        start = end
        end += row_len
