
import tensorflow as tf 
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Conv3D,
                                            Dense, Flatten,
                                            GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Input,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, Dropout, concatenate,
											UpSampling2D)
from tensorflow.python.keras import applications, regularizers
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import backend as K_B

def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME, use_global_average = False):
    '''
    Parameters
    ----------
    base_model: This is the pre-trained base model with which the non-trainable model is built

    Note: The term non-trainable can be confusing. The non-trainable-parametes are present only in this
    model. The other model (trianable model doesnt have any non-trainable parameters). But if you chose to 
    omit the bottlenecks due to any reason, you will be training this network only. (If you choose
    --omit_bottleneck flag). So please adjust the place in this function where I have intentionally made 
    certain layers non-trainable.

    Returns
    -------
    non_trainable_model: This is the model object which is the modified version of the base_model that has
    been invoked in the beginning. This can have trainable or non trainable parameters. If bottlenecks are
    created, then this network is completely non trainable, (i.e) this network's output is the bottleneck
    and the network created in the trainable is used for training with bottlenecks as input. If bottlenecks
    arent created, then this network is trained. So please use accordingly.
    '''
    # This post-processing of the deep neural network is to avoid memory errors
    x = (base_model.get_layer(BOTTLENECK_TENSOR_NAME))
    all_layers = base_model.layers
    for i in range(base_model.layers.index(x)):
        all_layers[i].trainable = False
    mid_out = base_model.layers[base_model.layers.index(x)]
    variable_summaries(mid_out.output)
    non_trainable_model = Model(base_model.input, mid_out.output)
    #non_trainable_model = Model(inputs = base_model.input, outputs = [x])
    
    # for layer in non_trainable_model.layers:
    #     layer.trainable = False
    
    return (non_trainable_model)
def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K_B.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x = x - tf.stack((tf.ones_like(x[0,:, :,:])*tf.constant(103.939),
                        tf.ones_like(x[1,:, :,:])*tf.constant(116.779)
                        ,tf.ones_like(x[2,:, :,:])*tf.constant(123.68)),axis=-1)
    
    else:
        # 'RGB'->'BGR'
        x = x[ :, :, ::-1]
        # Zero-center by mean pixel
        x = x - tf.stack((tf.ones_like(x[:,:,:,0])*tf.constant(103.939),
                        tf.ones_like(x[:,:,:,1])*tf.constant(116.779)
                        ,tf.ones_like(x[:,:,:,2])*tf.constant(123.68)),axis=-1)
    
    # x = 2*x/255

    return x

def backend_A(f, weights = None):

    

    x = Conv2D(512, 3, padding='same', dilation_rate=1,kernel_regularizer=regularizers.l2(0.01),  name="dil_A1")(model.output)
    x = Activation('relu')(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=1,kernel_regularizer=regularizers.l2(0.01),  name="dil_A2")(x)
    x = Activation('relu')(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=1,kernel_regularizer=regularizers.l2(0.01),  name="dil_A3")(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same', dilation_rate=1,kernel_regularizer=regularizers.l2(0.01),  name="dil_A4")(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same', dilation_rate=1,kernel_regularizer=regularizers.l2(0.01),  name="dil_A5")(x)
    x = Activation('relu')(x)
    x = Conv2D(64 , 3, padding='same', dilation_rate=1,kernel_regularizer=regularizers.l2(0.01),  name="dil_A6")(x)
    x = Activation('relu')(x)

    x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_A7")(x)

    model = Model(f.input, x, name = "Transfer_learning_model")
    return (model)

def backend_B(f, weights = None):
    
    
    x = Conv2D(512, 3, padding='same', dilation_rate=2, activation = 'relu', name="dil_B1")(f.output)
    # x = BatchNormalization(name='bn_b1')(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=2,activation = 'relu',name="dil_B2")(x)
    # x = BatchNormalization(name='bn_b2')(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=2,activation = 'relu',name="dil_B3")(x)
    # x = BatchNormalization(name='bn_b3')(x)
    x = Conv2D(256, 3, padding='same', dilation_rate=2, activation = 'relu',name="dil_B4")(x)
    # x = BatchNormalization(name='bn_b4')(x)
    x = Conv2D(128, 3, padding='same', dilation_rate=2, activation = 'relu',name="dil_B5")(x)
    # x = BatchNormalization(name='bn_b5')(x)
    x = Conv2D(64 , 3, padding='same', dilation_rate=2, activation = 'relu',name="dil_B6")(x)
    
    x = Conv2D(1, 1, padding='same', dilation_rate=1,   name="dil_B7")(x)
    model = Model(f.input, x, name = "Transfer_learning_model")
    return (model)

def backend_C(f, weights = None):

    x = Conv2D(512, 3, padding='same', dilation_rate=2,activation='relu', name="dil_C1")(f.output)
    x = Conv2D(512, 3, padding='same', dilation_rate=2,activation='relu', name="dil_C2")(x)