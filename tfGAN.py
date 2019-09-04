import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tqdm import tqdm

# TO DO: find a way to remove excessive tf warnings 

#writer = tf.summary.FileWriter("/home/stefan/tmp/gan/4")

# import data
image_path = "../image_data/bad_frames/*.png"

print("Importing Images from: {}\n".format(image_path))

x_train = []

for img_path in tqdm(glob.glob(image_path)):
    x_train.append(plt.imread(img_path))

x_train = np.array(x_train)

print("\n{} Images Imported!".format(len(x_train)))

# normalize the data
x_train = x_train / 255.

# define dictionary for parameters
params = dict(
    batch_size = 25,
    epochs = 10000,

    latent_dim = 100,
    height = 100,
    width = 100, 
    channels = 3,

    disc_learning_rate = 1e-4,
    gen_learning_rate= 0.00001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-07
)

# TO DO: make images greyscale # skimage.color.rgb2gray() completely removes the 'channels' axis, making it unsuitible for feeding into the tf networks

#x_train = rgb2gray(x_train) 

# Create tensorflow dataset and corresponding iterator with the imported data
with tf.name_scope("data"):
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(params['batch_size'])
    dataset = dataset.shuffle(len(x_train))
    dataset = dataset.repeat(count=None)
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()

# setup noise for z (returns a (50, 100) sample of gaussian noise)
def get_noise():
    with tf.name_scope("noise"):
        noise = tfp.distributions.Normal(tf.zeros(params['latent_dim']), tf.ones(params['latent_dim'])).sample(params['batch_size'])
    return noise

# define placeholders (x: image data, z: random latent vectors, y: labels) NOTE: UNUSED
'''
x = tf.compat.v1.placeholder(tf.float32, shape=[params['batch_size'], params['height'], params['width'], params['channels']])
z = tf.compat.v1.placeholder(tf.float32, shape=[params['batch_size'], params['latent_dim']])
y = tf.compat.v1.placeholder(tf.float64, shape=[params['batch_size'], 1])
'''

# setup function to return placeholder shape as np array
def get_shape(placeholder):
    temp = []
    for i in placeholder.shape.dims:
        temp.append(i.value)
    return np.asarray(temp)

# store shapes of placeholders NOTE: UNUSED
'''
x_shape = get_shape(x)
z_shape = get_shape(z)
y_shape = get_shape(y)
'''

# setup 2d convolution layer
def conv_2d_layer(input_data, num_input_channels, num_filters, filter_shape, activation, stride, padding, name):
    with tf.name_scope("conv2d"):

        # setup filter shape
        conv_filter_shape = [filter_shape[0], filter_shape[1],
                            num_input_channels, num_filters]
    
        # init weights and bias
        w = tf.compat.v1.get_variable(name=name+"_W", shape=conv_filter_shape, dtype=tf.float32, initializer=tf.initializers.truncated_normal)
        b = tf.compat.v1.get_variable(name=name+"_b", shape=[num_filters], dtype=tf.float32, initializer=tf.initializers.truncated_normal)

        # setup strides
        stride = [1, stride[0], stride[1], 1]

        # setup 2D convolution op
        layer_out = tf.nn.conv2d(input=input_data, filter=w, strides=stride, padding=padding)

        # add bias
        layer_out = layer_out + b

        # apply specified activation function
        if activation == "leaky_relu":
            layer_out = tf.nn.leaky_relu(layer_out)
        elif activation == "tanh":
            layer_out = tf.nn.tanh(layer_out)
        else:
            raise Exception('Error({}) - None or invalid activation function specified:({})'.format(name, activation))

        return layer_out

# function to determine the input_shape of the conv_2d_transpose op NOTE: UNUSED
def get_transpose_shape(output_shape, w, strides):
    output = np.ones(shape=output_shape)
    w = np.ones(shape=w.get_shape())
    #output = tf.compat.v1.placeholder(dtype=tf.float32, shape=output_shape) # PROBLEM: placeholders require feeding, find solution that doesnt use placeholders
    #w = tf.compat.v1.placeholder(dtype=tf.float32, shape=w.get_shape()) #
    transpose_shape = tf.nn.conv2d(output, w, strides=strides, padding='SAME')
    return transpose_shape


# setup 2d conv transposed layer
def conv_2d_transpose_layer(input_data, num_input_channels, num_filters, filter_shape, stride, padding, name):
    with tf.name_scope("deconv"):
        # setup filter shape
        conv_filter_shape = [filter_shape[0], filter_shape[1],
                            num_filters, num_input_channels] # NOTE: these two features a swapped in the cov2d_transpose operation for some reason

        input_shape = get_shape(input_data)

        output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, num_filters]

        # init weights and bias
        w = tf.compat.v1.get_variable(name=name+"_W", shape=conv_filter_shape, dtype=tf.float32, initializer=tf.initializers.truncated_normal)
        b = tf.compat.v1.get_variable(name=name+"_b", shape=[num_filters], dtype=tf.float32, initializer=tf.initializers.truncated_normal)

        # setup strides
        stride = [1, stride[0], stride[1], 1]

        # setup conv 2d transpose op
        layer_out = tf.nn.conv2d_transpose(value=input_data, filter=w, output_shape=output_shape, strides=stride, padding=padding)

        # add bias
        layer_out = layer_out + b

        # apply leaky relu
        layer_out = tf.nn.leaky_relu(layer_out)

        return layer_out

# setup average pooling layer
def avg_pooling(input_data, pool_shape, stride, padding):
    with tf.name_scope("avg_pool"):
        # setup kernal size
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        stride = [1, stride[0], stride[1], 1]

        # setup average pool op 
        layer_out = tf.nn.avg_pool(input_data, ksize=ksize, strides=stride, padding=padding)
        return layer_out

# setup fully connected layer op
def fc_layer(input_data, input_shape, num_units, activation, name):
    with tf.name_scope("fc"):
        # setup weights and bias
        w = tf.compat.v1.get_variable(name=name+"_W", shape=[input_shape[1], num_units], dtype=tf.float32, initializer=tf.initializers.truncated_normal)
        b = tf.compat.v1.get_variable(name=name+"_b", shape=[num_units], dtype=tf.float32, initializer=tf.initializers.truncated_normal)

        # setup matmul op
        layer_out = tf.matmul(input_data, w)
    
        # add bias
        layer_out = layer_out + b

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

# abstraction function to concat two tensors
def concatenate_layer(real_data, generated_data):
    layer_out = tf.concat(real_data, generated_data, 0)
    return layer_out

# --- define the Generative Adversarial Network ---
# ----- define Generator Network -----
def generator(z, name):
    with tf.compat.v1.variable_scope("generator"):
        print("-----{} Summary-----".format(name))

        # input: (batch_size, 100) (random latent vector)
        print("gen_input output shape:", get_shape(z))

        _, gen_fc1 = fc_layer(z, get_shape(z), (50 * 50 * 64), "leaky_relu", "gen_fc1") 
        # out: (batch_size 160000)
        print("gen_fc1 output shape:", get_shape(gen_fc1))

        gen_reshape = tf.reshape(gen_fc1, [-1 ,50, 50, 64])
        # out: (batch_size 50, 50, 64)
        print("gen_reshape output shape:", get_shape(gen_reshape))

        gen_conv1 = conv_2d_layer(gen_reshape, 64, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv1")
        # out: (batch_size 50, 50, 128)
        print("gen_conv1 output shape:", get_shape(gen_conv1))

        gen_conv_t1 = conv_2d_transpose_layer(gen_conv1, 128, 128, (4, 4), (2, 2), "SAME", "gen_conv_t1")
        # out: (batch_size 100, 100, 128)
        print("gen_conv_t1 output shape:", get_shape(gen_conv_t1))

        gen_conv2 = conv_2d_layer(gen_conv_t1, 128, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv2")
        # out: (batch_size 100, 100, 128)
        print("gen_conv2 output shape:", get_shape(gen_conv2))

        gen_conv3 = conv_2d_layer(gen_conv2, 128, 128, (5, 5), "leaky_relu", (1, 1), "SAME", "gen_conv3")
        # out: (batch_size 100, 100, 128)
        print("gen_conv3 output shape:", get_shape(gen_conv3))

        gen_conv4 = conv_2d_layer(gen_conv3, 128, 3, (7, 7), "tanh", (1, 1), "SAME", "gen_conv4")
        # out: (batch_size 100, 100, 3)
        print("gen_conv4 output shape:", get_shape(gen_conv4))

    return gen_conv4

# ----- define Discriminator Network -----
def discriminator(x, name, reuse=False):
    with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
        print("-----{} Summary-----".format(name))

        # input: (batch_size 100, 100, 3) concatenation of the real images with the generated images produced by the generator
        print("dis_input output shape:", get_shape(x))

        dis_conv1 = conv_2d_layer(x, 3, 128, (3, 3), "leaky_relu", (1, 1), "VALID", "dis_conv1")
        # out: (batch_size 98, 98, 128)
        print("dis_conv1 output shape:", get_shape(dis_conv1))

        dis_conv2 = conv_2d_layer(dis_conv1, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv2")
        # out: (batch_size 48, 48, 128)
        print("dis_conv2 output shape:", get_shape(dis_conv2))

        dis_conv3 = conv_2d_layer(dis_conv2, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv3")
        # out: (batch_size 23, 23, 128)
        print("dis_conv3 output shape:", get_shape(dis_conv3))

        dis_conv4 = conv_2d_layer(dis_conv3, 128, 128, (4, 4), "leaky_relu", (2, 2), "VALID", "dis_conv4")
        # out: (batch_size 10, 10, 128)
        print("dis_conv4 output shape:", get_shape(dis_conv4))

        dis_flatten = tf.reshape(dis_conv4, [-1, 10 * 10 * 128])
        # out: (batch_size 12800)
        print("dis_flatten output shape:", get_shape(dis_flatten))

        dis_dropout = tf.nn.dropout(dis_flatten, rate=0.4)
        # out: (batch_size 12800)
        print("dis_dropout output shape:", get_shape(dis_dropout))

        _, dis_output = fc_layer(dis_dropout, [-1, 10 * 10 * 128], 1, "sigmoid", "dis_output")
        # out: (batch_size 1) probability that the image being judged is real or generated
        print("dis_output output shape:", get_shape(dis_output))
    return dis_output

# create networks
samples = generator(z=get_noise(), name="Generator")
discrim = discriminator(x=tf.concat([x, samples], 0), name="Discriminator (real_score)")
gan_model = discriminator(x=samples, name="Discriminator (fake_score)", reuse=True)

# define the loss function
with tf.name_scope("d_loss"):
    d_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=discrim, labels=tf.concat([tf.ones([25,1]), tf.zeros([25,1])], 0))
    )

with tf.name_scope("a_loss"):
    a_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_model, labels=tf.zeros_like(gan_model))
    )

gen_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "generator")
disc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

# setup optimizers and update routines

# Discriminator update
with tf.name_scope("d_train"):
    d_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=params['disc_learning_rate'])
    d_opt = d_opt.minimize(d_loss)

# Generator update
with tf.name_scope("g_train"):
    g_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=params['gen_learning_rate'],
                                            beta1=params['beta1'],
                                            beta2=params['beta2'], 
                                            epsilon=params['epsilon'])
    g_opt = g_opt.minimize(a_loss)

#setup learning procedures
sess = tf.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

#writer.add_graph(sess.graph)
print("Graph added!")

# start training loop
a_losses = []
d_losses = []
for i in tqdm(range(params['epochs'])):
    _, _, l, ll = sess.run([g_opt, d_opt, d_loss, a_loss])
    print("Epoch: {} d_loss|a_loss ----- {}|{}".format(i, l, ll))
    d_losses.append(l)
    a_losses.append(ll)
    if i % 100 == 0:
        plt.imshow(sess.run(samples)[0])
        plt.savefig("{}_epochs.png".format(i), format='png')

# TO DO: find out why the only output for the generator is noise, maybe something isn't connected properly? Check tensorboard