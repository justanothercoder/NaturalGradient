import numpy as np
import matplotlib.pyplot as plt
import seaborn

def plot(params):

    for i, model in enumerate(params):
        with np.load('models/model_%s_val_error.npz' % model) as f:
            val_err = f['arr_0']
        
#        if model.find('nesterov') != -1:
#            style = '--'
#        elif model.find('div') != -1:
#            style = '-.'
#        else:
#            style = '-'
        style = '-'

        plt.plot(val_err, label=model, linestyle=style)

#params = [
#    'custom_momentum-1.0-0.9',
#    'custom_momentum-0.1-0.9',
#    'custom_momentum-0.01-0.9',
#    'custom_momentum-0.001-0.9',
#    'custom_momentum-0.1-0.5',
#    'custom_momentum-0.1-0.1',
#    ]

#params = [
#    'custom_momentum-1.0divk**1.0-0.9',
#    'custom_momentum-1.0divk**0.75-0.9',
#    'custom_momentum-1.0divk**0.5-0.9',
#    'custom_nesterov_momentum-1.0divk**1.0-0.9',
#    'custom_nesterov_momentum-1.0divk**0.75-0.9',
#    'custom_nesterov_momentum-1.0divk**0.5-0.9',
#]

#params = [
#    'custom_rmsprop_0.01-0.9',
#    'custom_rmsprop_0.01-0.6',
#    'custom_rmsprop_0.01-0.3',
#    'custom_rmsprop_0.01-0.1'
#]

params = [
    'custom_adam_0.01_0.9_0.999',
#    'custom_adam_0.01_0.5_0.999',
#    'custom_adam_0.01_0.1_0.999',
#                               
#    'custom_adam_0.01_0.9_0.5',
#    'custom_adam_0.01_0.9_0.1',
#
#    'custom_adam_0.1_0.9_0.999',
#    'custom_adam_1.0_0.9_0.999',
#    'custom_adam_10.0_0.9_0.999',
]

plot(params)

plt.title('Validation error/epoch')    
plt.legend()
plt.show()

