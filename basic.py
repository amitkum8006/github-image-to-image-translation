import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation
#Training PArams
learning_rate = 0.0002
batch_size = 32
epochs = 100000

#Network params
image_dimension = 784 #img sz is 28x28
#Discriminator Nodes
H_dim = 128

def xavier_init(shape):
  return tf.random_normal(shape = shape, stddev= 1./tf.sqrt(shape[0]/2.0))
tf.__version__
'1.2.0'
#define placeholders for external input

X_A = tf.placeholder(tf.float32, shape = [None, image_dimension])
X_B = tf.placeholder(tf.float32, shape = [None, image_dimension])
#define placeholders for external input

X_A = tf.placeholder(tf.float32, shape = [None, image_dimension])
X_B = tf.placeholder(tf.float32, shape = [None, image_dimension])
# Define weights and biases for dictionaries for Discriminator A

Disc_A_W = { "disc_H" : tf.Variable(xavier_init([image_dimension, H_dim])),
             "disc_final": tf.Variable(xavier_init([H_dim, 1]))}

Disc_A_Bias = { "disc_H" : tf.Variable(xavier_init([H_dim])),
             "disc_final": tf.Variable(xavier_init([1]))}


# Define weights and biases for dictionaries for Discriminator B

Disc_B_W = { "disc_H" : tf.Variable(xavier_init([image_dimension, H_dim])),
             "disc_final": tf.Variable(xavier_init([H_dim, 1]))}

Disc_B_Bias = { "disc_H" : tf.Variable(xavier_init([H_dim])),
             "disc_final": tf.Variable(xavier_init([1]))}

# Define weights and biases for dictionaries for Generator transforming A to B

Gen_AB_W = { "Gen_H" : tf.Variable(xavier_init([image_dimension, H_dim])),
             "Gen_final": tf.Variable(xavier_init([H_dim, image_dimension]))} #784 due to output dimension of the image

Gen_AB_Bias = { "Gen_H" : tf.Variable(xavier_init([H_dim])),
             "Gen_final": tf.Variable(xavier_init([image_dimension]))}


# Define weights and biases for dictionaries for Generator transforming B to A

Gen_BA_W = { "Gen_H" : tf.Variable(xavier_init([image_dimension, H_dim])),
             "Gen_final": tf.Variable(xavier_init([H_dim, image_dimension]))}

Gen_BA_Bias = { "Gen_H" : tf.Variable(xavier_init([H_dim])),
             "Gen_final": tf.Variable(xavier_init([image_dimension]))}
