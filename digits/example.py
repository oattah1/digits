'''
Example TensorFlow module
'''
import tempfile
# Import the tensorflow module so that its definitions are in scope
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Define a function that runs an example tensorflow computation and returns the result
# The computation is a simple constant matrix multiplication
def softmax_digits():
   # lets first load the mnist data
    # one_hot means that we are representing the digits 0-9 as 10-element vectors
    # where exactly one element is 1 and all the others are 0.
    # e.g. 0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #      3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    tmpdir = tempfile.mkdtemp()
    mnist = input_data.read_data_sets(tmpdir, one_hot=True)

    # This is an input tensor to the computation graph. It has two dimensions,
    # one of them has unknown size and the other a size of 784 entries, one for each pixel.
    # The dimension of unknown size means that we can pass in any number of images each comprising
    # 784 pixels.
    x = tf.placeholder(tf.float32, [None, 784])

    # The weights of our network: we have 10 outputs for each of 784 input pixels.
    W = tf.Variable(tf.zeros([784, 10]))
    # The bias of our network: 1 element for each digit 0-9
    b = tf.Variable(tf.zeros([10]))

    # Note: Our weights and biases are specified as variables, so they have a definite
    # value within our computation graph which can be updated by tensorflow calculations.
    # Contrast with the placeholder which is an input we supply for each calculation.

    # The pieces of the computation graph we've defined so far compose our model:
    y = tf.matmul(x, W) + b

    # To check our model we need to test it against correct labels
    # Again, each answer is represented as a 10-element vector.
    # When we run a training or validation batch we supply the training data in `x`
    # and the labels we expect in `y_`.
    y_ = tf.placeholder(tf.float32, [None, 10])

    # We evaluate our model by computing its cross-entropy with the correct labels
    # `y` is a tensor of shape [None, 10] just like `y_`.
    # 1. softmax normalization of our current guess
    # 2. cross-entropy is multiply, sum across the second axis and negate
    # 3. error is average over all the examples (the first axis)
    #
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), axis=[1]))
    #
    # However that is numerically unstable so we do this instead:
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Now we finish defining our computational graph by specifying how we want to run the
    # optimization step. We use gradient descent to minimize cross_entropy, our error function.
    train_step = train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # having defined our computational graph, let's run it, print out the result, and return it
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
     # training phase. We use batches of 100 images, 1000 times.
    for _ in range(1000):
        # fetch a batch of test images and their labels
        batch_xs, batch_ys = mnist.train.next_batch(100);
        # run the computational graph, making sure to provide the inputs
        # we promised TensorFlow we would provide
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
     # validation phase. we use a variant of the computational graph here
    # since each answer label is a one_hot vector, we just find the indexes
    # of the predicted and correct label and compare them for equality.
    #
    # The accuracy is just the average number of correct answers we got.
    # Note that we have to cast the true/false answer in correct_prediction to
    # a floating point number.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # We test against all of the validation data
    result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Result of tensorflow computation: " + str(result))
    return result
