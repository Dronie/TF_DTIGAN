import tensorflow as tf
import numpy as np

params = dict(
    learning_rate = 0.5,
    batch_size = 50,
    epochs = 10,
    z_dim = 50,
    latent_dim = 100
)

# define placeholders (x: image data, z: random latent vectors, y: labels)
x = tf.placeholder(tf.float32, shape=[params['batch_size'], 100, 100, 3])
z = tf.placeholder(tf.float32, shape=[params['batch_size'], params['z_dim']])
y = tf.placeholder(tf.float64, shape=[params['batch_size'] * 2, 1])

def get_shape(placeholder):
    temp = []
    for i in placeholder.shape.dims:
        temp.append(i.value)
    return np.asarray(temp)

x_shape = get_shape(x)
z_shape = get_shape(z)
y_shape = get_shape(y)

# setup 2d convolution layer
def conv_2d_layer(input_data, num_input_channels, num_filters, filter_shape, activation, stride, padding, name):
    # setup filter shape
    conv_filter_shape = [filter_shape[0], filter_shape[1],
                        num_input_channels, num_filters]
    
    # init weights and bias
    w = tf.get_variable(name+"_W", conv_filter_shape, tf.float32, tf.initializers.truncated_normal)
    b = tf.get_variable(name+"_b", [num_filters], tf.float32, tf.initializers.truncated_normal)

    # setup strides
    stride = [1, stride[0], stride[1], 1]

    # setup 2D convolution op
    layer_out = tf.nn.conv2d(input_data, w, stride, padding=padding)

    # add bias
    layer_out += b

    # apply specified activation function
    if activation == "leaky_relu":
        layer_out = tf.nn.leaky_relu(layer_out)#
    elif activation == "tanh":
        layer_out = tf.nn.tanh(layer_out)
    else:
        raise Exception('Error({}) - None or invalid activation function specified:({})'.format(name, activation))

    return layer_out

# setup 2d conv transposed layer
def conv_2d_transpose_layer(input_data, num_input_channels, num_filters, filter_shape, stride, padding, name):
    # setup filter shape
    conv_filter_shape = [filter_shape[0], filter_shape[1],
                        num_filters, num_input_channels] # NOTE: these two features a swapped in the cov2d_transpose operation for some reason
    
    # init weights and bias
    w = tf.get_variable(name=name+"_W", shape=conv_filter_shape, dtype=tf.float32, initializer=tf.initializers.truncated_normal)
    b = tf.get_variable(name=name+"_b", shape=[num_filters], dtype=tf.float32, initializer=tf.initializers.truncated_normal)

    # setup strides
    stride = [1, stride[0], stride[1], 1]

    # setup conv 2d transpose op
    layer_out = tf.nn.conv2d_transpose(input_data, w, stride, padding=padding)

    # add bias
    layer_out += b

    # apply leaky relu
    layer_out = tf.nn.leaky_relu(layer_out)

    return layer_out

# setup average pooling layer
def avg_pooling(input_data, pool_shape, stride, padding):
    # setup kernal size
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    stride = [1, stride[0], stride[1], 1]

    # setup average pool op 
    layer_out = tf.nn.avg_pool(input_data, ksize=ksize, strides=stride, padding=padding)
    return layer_out

# setup fully connected layer op
def fc_layer(input_data, input_shape, num_units, activation, name):
    # setup weights and bias
    w = tf.get_variable(name=name+"_W", shape=list(np.concatenate(input_shape, num_units)), dtype=tf.float32, initializer=tf.initializers.truncated_normal)
    b = tf.get_variable(name=name+"_b", shape=[num_units], dtype=tf.float32, initializer=tf.initializers.truncated_normal)

    # setup matmul op
    layer_out = tf.add(tf.matmul(input_data, w), b)

    # apply specified activtion function
    if activation == "relu":
        act_layer_out = tf.nn.relu(layer_out)
    elif activation == "softmax":
        act_layer_out = tf.nn.softmax(layer_out)
    elif activation == "leaky_relu":
        act_layer_out = tf.nn.leaky_relu(layer_out)
    elif activation == "sigmoid":
        act_layer_out = tf.nn.sigmoid(layer_out)
    else:
        raise Exception('Error({}) - None or invalid activation function specified:({})'.format(name, activation))
    
    return layer_out, act_layer_out

def concatenate_layer(real_data, generated_data):
    layer_out = tf.concat(real_data, generated_data, 0)
    return layer_out

# --- define the Generative Adversarial Network ---
def generator(z):
    with tf.compat.v1.variable_scope("generator"):
        # input: (None, 100) (random latent vector)
        _, gen_fc1 = fc_layer(z, z_shape, (50 * 50 * 64), "leaky_relu", "gen_fc1")
        # out: (None, 160000)
        gen_reshape = tf.reshape(gen_fc1, [-1 ,50, 50, 64])
        # out: (None, 50, 50, 64)
        gen_conv1 = conv_2d_layer(gen_reshape, 64, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv1")
        # out: (None, 50, 50, 128)
        gen_conv_t1 = conv_2d_transpose_layer(gen_conv1, 128, 128, (4, 4), (2, 2), "SAME", "gen_conv_t1")
        # out: (None, 100, 100, 128)
        gen_conv2 = conv_2d_layer(gen_conv_t1, 128, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv2")
        # out: (None, 100, 100, 128)
        gen_conv3 = conv_2d_layer(gen_conv2, 128, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv3")
        # out: (None, 100, 100, 128)
        gen_conv4 = conv_2d_layer(gen_conv3, 128, 3, (7, 7), "tanh", (1, 1), "SAME", "gen_conv4")
        # out: (None, 100, 100, 3)
    return gen_conv4

def discriminator(x, reuse=False):
    with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
        # input: (None, 100, 100, 3) concatenation of the real images with the generated images produced by the generator
        dis_conv1 = conv_2d_layer(x, x_shape, 128, (3, 3), "leaky_relu", (1, 1), "VALID", "dis_conv1")
        # out: (None, 98, 98, 128)
        dis_conv2 = conv_2d_layer(dis_conv1, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv2")
        # out: (None, 48, 48, 128)
        dis_conv3 = conv_2d_layer(dis_conv2, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv3")
        # out: (None, 23, 23, 128)
        dis_conv4 = conv_2d_layer(dis_conv3, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv4")
        # out: (None, 10, 10, 128)
        dis_flatten = tf.reshape(dis_conv4, [-1, 10 * 10 * 128])
        # out: (None, 12800)
        dis_dropout = tf.nn.dropout(dis_flatten, rate=0.4)
        dis_output = fc_layer(dis_dropout, [-1 , 10 * 10 * 128], 1, "sigmoid", "dis_output")
        # out: (None, 1) probability that the image being judged is real or generated
    return dis_output

samples = generator(z)
real_score = discriminator(x)
fake_score = discriminator(samples, reuse=True)

# define a loss function
loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(real_score))+
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score))
)

