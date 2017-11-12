import tensorflow as tf
import numpy
import settings

rng = numpy.random

def linear_regression(feature_matrix, label_vector):
    # Training Data
    train_X = numpy.asarray(feature_matrix)
    train_Y = numpy.asarray(label_vector)

    n_samples = train_X.shape[0]

    with tf.device(settings.device):
        # tf Graph Input
        X = tf.placeholder("float")
        Y = tf.placeholder("float")

        # Set model weights
        W = tf.Variable(rng.randn(), name="weight")
        b = tf.Variable(rng.randn(), name="bias")

        # Construct a linear model
        pred = tf.add(tf.multiply(X, W), b)

        # Mean squared error
        cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
        # Gradient descent
        #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
        optimizer = tf.train.GradientDescentOptimizer(settings.learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
#   config=tf.ConfigProto(log_device_placement=True)
    config=tf.ConfigProto(intra_op_parallelism_threads=settings.NUM_THREADS)
    with tf.Session(config=config) as sess:
        sess.run(init)

        # Fit all training data
        for epoch in range(settings.training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})
