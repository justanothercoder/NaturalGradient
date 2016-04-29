import numpy as np
import matplotlib.pyplot as plt
import seaborn

params = [
    'custom_momentum-1.0-0.9',
    'custom_momentum-0.1-0.9',
    'custom_momentum-0.01-0.9',
    'custom_momentum-0.001-0.9',
    'custom_momentum-0.1-0.5',
    'custom_momentum-0.1-0.1',
    'custom_momentum-1.0divk**1.0-0.9',
    'custom_momentum-1.0divk**0.75-0.9',
    'custom_momentum-1.0divk**0.5-0.9',
    ]
for i, model in enumerate(params):
    with np.load('model_%s_val_error.npz' % model) as f:
        val_err = f['arr_0']
    plt.plot(val_err, label=model)

plt.title('Validation error/epoch')    
plt.legend()
plt.show()

