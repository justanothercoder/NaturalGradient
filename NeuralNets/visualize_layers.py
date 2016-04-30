import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math

model = 'adam_reg'

with np.load('models/model_%s.npz' % model) as f:
    param_values = [f['arr_%d' % j] for j in range(len(f.files))]

n_images = 16
for i in range(n_images):
    img = param_values[0][:, i].reshape(28, 28)

    print img.shape
    plt.subplot(math.floor(math.sqrt(n_images)), math.ceil(n_images / math.sqrt(n_images)), i+1)
    plt.axis('off')
#    plt.imshow(img, cmap='seismic')
    plt.imshow(img, cmap='Greys')

plt.show()

    #lasagne.layers.set_all_param_values(network, param_values)
