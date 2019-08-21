import tensorflow as tf

params = dict(
    learning_rate = 0.5,
    batch_size = 50,
    epochs = 10,
    latent_dim = 100
)

# define placeholders (x: image data, z: random latent vectors, y: labels)
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
z = tf.placeholder(tf.float32, shape=[None, params['latent_dim']])
y = tf.placeholder(tf.float64, shape=[None, params['batch_size'] * 2, 1])

# setup 2d convolution layer
def conv_2d_layer(input_data, num_input_channels, num_filters, filter_shape, activation, stride, padding, name):
    # setup filter shape
    conv_filter_shape = [filter_shape[0], filter_shape[1],
                        num_input_channels, num_filters]
    
    # init weights and bias
    w = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03, name=name+"_W"))
    b = tf.Variable(tf.truncated_normal([num_filters], name=name+"_b"))

    # setup strides
    stride = [1, stride[0], stride[1], 1]

    # setup 2D convolution op
    layer_out = tf.nn.conv_2d(input_data, w, stride, padding=padding)

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
                        num_filters, num_input_channels] # NOTE: these two features a swapped in cov2d_transpose operation for some reason
    
    # init weights and bias
    w = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03, name=name+"_W"))
    b = tf.Variable(tf.truncated_normal([num_filters], name=name+"_b"))

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
    w = tf.Variable(tf.truncated_normal([input_shape, num_units], stddev=0.03, name=name+"_W"))
    b = tf.Variable(tf.truncated_normal([num_units], stddev=0.03, name=name+"_b"))

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

# --- define the generator network --- 
# input: (None, 100)
_, gen_fc1 = fc_layer(z, [None, params['latent_dim']], (50 * 50 * 64), "leaky_relu", "gen_fc1")
# out: (None, 160000)
gen_reshape = tf.reshape(gen_fc1, [-1 ,50, 50, 64])
# out: (None, 50, 50, 64)
gen_conv1 = conv_2d_layer(gen_fc1, 64, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv1")
# out: (None, 50, 50, 128)
gen_conv_t1 = conv_2d_transpose_layer(gen_conv1, 128, 128, (4, 4), (2, 2), "SAME", "gen_conv_t1")
# out: (None, 100, 100, 128)
gen_conv2 = conv_2d_layer(gen_conv_t1, 128, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv2")
# out: (None, 100, 100, 128)
gen_conv3 = conv_2d_layer(gen_conv2, 128, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv3")
# out: (None, 100, 100, 128)
gen_conv4 = conv_2d_layer(gen_conv3, 128, 3, (7, 7), "tanh", (1, 1), "SAME", "gen_conv4")
# out: (None, 100, 100, 3)

# --- define discriminator network ---
# input: (None, 100, 100, 3)
dis_conv1 = conv_2d_layer(x, [None, 100, 100, 3], 128, (3, 3), "leaky_relu", (1, 1), "VALID", "dis_conv1")
# out: (None, 98, 98, 128)
dis_conv2 = conv_2d_layer(dis_conv1, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv2")
# out: (None, 48, 48, 128)
dis_conv3 = conv_2d_layer(dis_conv2, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv3")
# out: (None, 23, 23, 128)
dis_conv4 = conv_2d_layer(dis_conv3, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv4")
# out: (None, 10, 10, 128)
dis_flatten = tf.reshape(-1 , 10 * 10 * 128)
# out: (None, 12800)
dis_dropout = tf.nn.dropout(dis_flatten, rate=0.4)
dis_output = fc_layer(dis_dropout, [-1 , 10 * 10 * 128], 1, "sigmoid", "dis_output")
# out: (None, 1)

