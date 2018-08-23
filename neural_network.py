# Feed Forward Network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Layer Anzahl
layers = [
    ['inputlayer', 784],
    ['hidden1', 500],
    ['hidden2', 500],
    ['outputlayer', 100]
]

model_input_nodes = layers[0][1]
model_classes = 10
model_batch_size = 150

X = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# Neural Net

def nn_model(X, classes, layers):
    maxlayers = len(layers)-1
    for l in range(0,maxlayers+1):
        m = l+1
        if l != maxlayers:
            vars()[layers[l][0]] ={
                'weights': tf.Variable(tf.truncated_normal([layers[l][1], layers[m][1]], stddev=0.1)),
                'biases': tf.Variable(tf.truncated_normal([layers[m][1]]))
                }
        else:
            vars()[layers[l][0]] ={
                'weights': tf.Variable(tf.truncated_normal([layers[l][1], classes], stddev=0.1)),
                'biases': tf.Variable(tf.truncated_normal([classes]))
                }

    for l in range(0,maxlayers+1):
        if l == 0:
            layer_sum = tf.add(tf.matmul(X, vars()[layers[l][0]]['weights']), vars()[layers[l][0]]['biases'])
            layer_sum = tf.nn.relu(layer_sum)
        elif l != maxlayers:
            layer_sum = tf.add(tf.matmul(layer_sum, vars()[layers[l][0]]['weights']), vars()[layers[l][0]]['biases'])
            layer_sum = tf.nn.relu(layer_sum)
        else:
            layer_sum = tf.add(tf.matmul(layer_sum, vars()[layers[l][0]]['weights']), vars()[layers[l][0]]['biases'])

    return layer_sum
                    

def nn_train(X, classes, layers, epochs=1, batch_size=100):
    pred = nn_model(X, classes, layers)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0.0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch ', epoch, ' of ', epochs, ' with loss: ', epoch_loss)

        correct_result = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_result, 'float'))
        print('Acc: ', accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))

# Test Model
epochs_list = [5, 10, 20, 30]

for epochs in epochs_list:
    nn_train(X, model_classes, layers, epochs, model_batch_size)



# Bestes Ergebis Acc ca. 98,2 %
# 3 Layer, 10 epochs und Batch Size 150
layers = [
    ['inputlayer', 784],
    ['hidden1', 500],
    ['hidden2', 500],
    ['outputlayer', 100]
]
print('best model:')
nn_train(X, 10, layers, 10, 150)