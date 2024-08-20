import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ConvBlock(layers.Layer):
    
    def __init__(self, num_filters, kernel_size, strides=1, data_format='channels_last'):
        
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        
        self.conv = layers.Conv2D(num_filters,
                                            kernel_size,
                                            strides,
                                            padding = ('same' if strides == 1 else 'valid'), 
                                            data_format=data_format, 
                                            use_bias=False)
        self.batchnorm = layers.BatchNormalization(
                            axis = -1 if data_format == "channels_last" else 1,
                            momentum=0.99,
                            #epsilon=0.01)
                            epsilon=1e-05)
        self.leaky_relu = layers.LeakyReLU(alpha=0.1)

    def call(self, x, training=False):
        
        if self.strides > 1:
            x = self._fixed_padding(x, self.kernel_size)
            
        output = self.conv(x)
        output = self.batchnorm(output, training=training)
        output = self.leaky_relu(output)
        
        return output
    
    def _fixed_padding(self, inputs, kernel_size, mode='CONSTANT'):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if self.data_format == 'channels_first':
            padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
        else:
            padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode=mode)

        return padded_inputs    

class DarknetBlock(layers.Layer):
    
    def __init__(self, num_filters, data_format='channels_last'):
        
        super(DarknetBlock, self).__init__()

        self.conv1 = ConvBlock(num_filters, 1, data_format=data_format)
        self.conv2 = ConvBlock(num_filters*2, 3, data_format=data_format)
        self.add = layers.Add()
    
    def call(self, x, training=False):

        shortcut = x
        output = self.conv1(x,training=training)
        output = self.conv2(output,training=training)        
        output = self.add([output, shortcut])
            
        return output

inputs = tf.keras.Input(shape=(256,256,3), name='img')
output = ConvBlock(32,3,strides=1)(inputs)
output = ConvBlock(64,3,strides=2)(output)

output = DarknetBlock(32)(output)

output = ConvBlock(128,3,strides=2)(output)
for _ in range(2):
    output = DarknetBlock(64)(output)

output = ConvBlock(256,3,strides=2)(output)
for _ in range(8):
    output = DarknetBlock(128)(output)

output = ConvBlock(512,3,strides=2)(output)
for _ in range(8):
    output = DarknetBlock(256)(output)

output = ConvBlock(1024,3,strides=2)(output)  
for _ in range(4):
    output = DarknetBlock(512)(output)

output = layers.GlobalAvgPool2D()(output)
output = layers.Reshape((1,1,1024))(output)
output = layers.Conv2D(1000,1)(output)

model = tf.keras.Model(inputs=inputs, outputs=output)

def load_weights(model, weights_file):
    """
    Loads and converts pre-trained weights.
    :param model: Keras model
    :param weights_file: name of the binary file.
    :return total_params: if load successfully end else -1
    """
                               
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)
    

    ptr = 0
    i = 0
    total_params = 0

    var_list = model.variables
    
    var_name_list = ['/'.join(x.name.split('/')[:-1]) for x in var_list]

    print(len(model.variables))
    print(len(model.trainable_variables))
    
    assert len(var_list) - 105 != len(model.trainable_variables), 'list length is wrong'

    while i < len(model.trainable_variables):
        var1 = var_list[i]
        var2 = var_list[i + 1]

        print("%d - var1 : %s (%s)" %(i, var1.name.split('/')[-2], var1.name.split('/')[-1]))
        print("%d - var2 : %s (%s)" %(i, var2.name.split('/')[-2], var2.name.split('/')[-1]))        
        
        # do something only if we process conv layer        
        if 'conv2' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                
                # load batch norm's gamma and beta params
                # beta bias
                # gamma kernel                
                gamma, beta = var_list[i + 1:i + 3]
                
                # Find mean and variance of the same name  
                layer_name = '/'.join(gamma.name.split('/')[:-1])
                mean_index = i + 3
                mean_index += var_name_list[i+3:].index(layer_name)
                mean, var = var_list[mean_index:mean_index+2] 

                batch_norm_vars = [beta, gamma, mean, var]
                
                for batch_norm_var in batch_norm_vars:
                    shape = batch_norm_var.shape.as_list()
                    num_params = np.prod(shape)
                    batch_norm_var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                  
                    batch_norm_var.assign(batch_norm_var_weights, name=batch_norm_var.name)
                    #assign_ops.append(
                    #    tf.compat.v1.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 2
            elif 'conv2' in var2.name.split('/')[-2]:
                # load biases
                print("%d - var2 : %s" %(i, var2.name.split('/')[-2]))
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params

                bias.assign(bias_weights, name=bias.name)

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
        
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            var1.assign(var_weights, name=var1.name)
            i += 1
            
        total_params = ptr
            
    
    
    return total_params if total_params == weights.shape else -1

