# keras-attn_aug_cnn
Extension of the `Attention Augmented Convolutional Networks` paper for hacky 1-D convolution operation implementation.

## Properties

```
depth_k | filters, depth_v | filters,  Nh | depth_k, Nh | filters-depth_v
```

## 1-D CNN

```

from aug_attn import *
from keras.layers import Input
from keras.models import Model

ip = Input(shape=(32, 10))
cnn1 = Conv1D(filters = 10, kernel_size=3, strides=1,padding='same')(ip)
x = augmented_conv1d(cnn1, shape = (32, 10), filters=20, kernel_size=5,
                     strides = 1,
                     padding = 'causal', # if causal convolution is needed
                     depth_k=4, depth_v=4,  
                     num_heads=4, relative_encodings=True)

# depth_k | filters, depth_v | filters,  Nh | depth_k, Nh | filters-depth_v

model = Model(ip, x)
model.summary()

x = tf.ones((1, 32, 10))
print(x.shape)
y = model(x)
print(y.shape)

```




```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_15 (InputLayer)           (None, 32, 10)       0                                            
__________________________________________________________________________________________________
conv1d_41 (Conv1D)              (None, 32, 10)       310         input_15[0][0]                   
__________________________________________________________________________________________________
conv1d_43 (Conv1D)              (None, 32, 12)       132         conv1d_41[0][0]                  
__________________________________________________________________________________________________
reshape_52 (Reshape)            (None, 32, 1, 12)    0           conv1d_43[0][0]                  
__________________________________________________________________________________________________
attention_augmentation2d_13 (At (None, 32, 1, 4)     64          reshape_52[0][0]                 
__________________________________________________________________________________________________
reshape_53 (Reshape)            (None, 32, 4)        0           attention_augmentation2d_13[0][0]
__________________________________________________________________________________________________
conv1d_42 (Conv1D)              (None, 32, 16)       816         conv1d_41[0][0]                  
__________________________________________________________________________________________________
conv1d_44 (Conv1D)              (None, 32, 4)        20          reshape_53[0][0]                 
__________________________________________________________________________________________________
reshape_51 (Reshape)            (None, 32, 1, 16)    0           conv1d_42[0][0]                  
__________________________________________________________________________________________________
reshape_54 (Reshape)            (None, 32, 1, 4)     0           conv1d_44[0][0]                  
__________________________________________________________________________________________________
concatenate_13 (Concatenate)    (None, 32, 1, 20)    0           reshape_51[0][0]                 
                                                                 reshape_54[0][0]                 
__________________________________________________________________________________________________
reshape_55 (Reshape)            (None, 32, 20)       0           concatenate_13[0][0]             
==================================================================================================
Total params: 1,342
Trainable params: 1,342
Non-trainable params: 0
__________________________________________________________________________________________________
(1, 32, 10)
(1, 32, 20)
```




## 2-D CNN


```
from aug_attn import *
from keras.layers import Input
from keras.models import Model

ip = Input(shape=(32, 32, 10))
cnn1 = Conv2D(filters = 10, kernel_size=3, strides=1,padding='same')(ip)
x = augmented_conv2d(cnn1, filters=20, kernel_size=5, # shape parameter is not needed
                     strides = 1,
                     depth_k=4, depth_v=4,  # padding is by default, same
                     num_heads=4, relative_encodings=True)

# depth_k | filters, depth_v | filters,  Nh | depth_k, Nh | filters-depth_v

model = Model(ip, x)
model.summary()

x = tf.ones((1, 32, 32, 10))
print(x.shape)
y = model(x)
print(y.shape)
```





```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_16 (InputLayer)           (None, 32, 32, 10)   0                                            
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 10)   910         input_16[0][0]                   
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 32, 32, 12)   132         conv2d_11[0][0]                  
__________________________________________________________________________________________________
attention_augmentation2d_14 (At (None, 32, 32, 4)    126         conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 16)   4016        conv2d_11[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 32, 32, 4)    20          attention_augmentation2d_14[0][0]
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 32, 32, 20)   0           conv2d_12[0][0]                  
                                                                 conv2d_14[0][0]                  
==================================================================================================
Total params: 5,204
Trainable params: 5,204
Non-trainable params: 0
__________________________________________________________________________________________________
(1, 32, 32, 10)
(1, 32, 32, 20)
```

# Implementations

* https://github.com/titu1994/keras-attention-augmented-convs
* https://github.com/gan3sh500/attention-augmented-conv
* https://github.com/leaderj1001/Attention-Augmented-Conv2d