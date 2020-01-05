import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

from config import config
from handleData import LoadData


# dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
x, y = mnist.train.next_batch(5)
print(type(x), x)
print(type(y), y)
loadData = LoadData()
train_text = loadData.train_text
valid_text = loadData.valid_text

# tf Graph input
X = tf.placeholder("float", [None, config.timesteps, config.num_input])
Y = tf.placeholder("float", [None, config.num_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([config.num_input, config.num_hidden])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([config.num_hidden, config.num_classes]))
}
biases = {
    # (128,)
    'in': tf.Variable(tf.random_normal([config.num_hidden])),
    # (10,)
    'out': tf.Variable(tf.random_normal([config.num_classes]))
}

def RNN(x, weights, biases):
# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, timesteps, n_input)
# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, timesteps, 1)

    # hidden layer for input to cell
    x = tf.reshape(x, [-1, config.num_input])  # (128*28, 28) 3D->2D
    x_in = tf.matmul(x, weights['in']) + biases['in']  # (128*28, 128)
    x_in = tf.reshape(x_in, [-1, config.timesteps, config.num_hidden])  # (128, 28, 128) 2D->3D

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(config.num_hidden, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(config.batch_size, dtype=tf.float32)

    # Get lstm cell output
    #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)  # timesteps在第二维所以是false

    # hidden layer for output as the final result
    #results = tf.matmul(states[1], weights['out']) + biases['out']
    #return results
    # Linear activation, using rnn inner loop last output
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

print('start training---')

# Start training
with tf.Session() as sess:
    # Run the initializer
    print('running the initializer')
    sess.run(init)

    for step in range(1, config.training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(config.batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape([config.batch_size, config.timesteps, config.num_input])
        # Run optimization op (backprop)
        sess.run([train_op], feed_dict={X: batch_x, Y: batch_y})
        if step % config.display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
print("Optimization Finished!")
'''
# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
'''