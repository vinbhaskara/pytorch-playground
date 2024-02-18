import numpy as np

_COLORS_ = ["#f5962a", "#117bbf"]  # orange (class 0), blue (class 1)
_PI_ = 3.1416


def gen_spiral(delta_t, num, noise=0):
    i = np.arange(num).astype(np.float32)
    r = i / num * 5
    t = 1.75 * i / num * 2 * _PI_ + delta_t
    x = r * np.sin(t) + np.random.uniform(-1, 1, t.shape) * noise
    y = r * np.cos(t) + np.random.uniform(-1, 1, t.shape) * noise
    x = x.reshape(num, -1)
    y = y.reshape(num, -1)
    return np.concatenate([x, y], axis=1)


def classify_spiral_data(num_samples, noise=0, random_seed=None, shuffle=False):
    """
    Example:
        >> inputs, labels, colors = classify_spiral_data(100, 0.0)
        >> plt.scatter(inputs[:, 0], inputs[:, 1], c=colors)
        >> plt.show()
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = num_samples // 2
    inputs = []
    labels = []

    inputs.append(gen_spiral(0, n, noise=noise))  # positive examples (blue - class 1)
    labels.extend([1] * n)

    inputs.append(
        gen_spiral(_PI_, num_samples - n, noise=noise)
    )  # negative examples (orange - class 0)
    labels.extend([0] * (num_samples - n))

    inputs = np.concatenate(inputs, axis=0).astype(np.float32)
    labels = np.array(labels).astype(np.int32)
    colors = [_COLORS_[i] for i in labels]

    if shuffle:
        indices = np.arange(inputs.shape[0]).astype(np.int32)
        np.random.shuffle(indices)
        inputs = inputs[indices]
        labels = labels[indices]
        colors = [colors[i] for i in indices.tolist()]

    return inputs, labels, colors
