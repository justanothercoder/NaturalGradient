import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math

#model = 'momentum_reg_denoising100'
#model = 'adam_reg_denoising100'

#model = 'adam_sparse_7.0_not_denoising'
model = 'svrg_100.0_squared_error'

with np.load('models/model_%s.npz' % model) as f:
    param_values = [f['arr_%d' % j] for j in range(len(f.files))]

    print param_values[2].shape

#n_images = 500
n_images = 300
#n_images = 25
for i in range(n_images):
    img = param_values[0][:, i].reshape(28, 28)

    print img.shape
    plt.subplot(math.floor(math.sqrt(n_images)), math.ceil(n_images / math.sqrt(n_images)), i+1)
    plt.axis('off')
#    plt.imshow(img, cmap='seismic')
    plt.imshow(img, cmap='Greys')

plt.show()
