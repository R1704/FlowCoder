import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d, n=10000):
    p = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            p[k, 2 * i] = np.sin(k / denominator)
            p[k, 2 * i + 1] = np.cos(k / denominator)
    return p


p = get_positional_encoding(seq_len=4, d=4, n=100)
print(p)

P = get_positional_encoding(seq_len=100, d=512, n=10000)
cax = plt.matshow(P)
plt.gcf().colorbar(cax)
plt.show()

