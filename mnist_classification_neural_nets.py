import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
	mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)

	#55,000 training data 10,000 test data 5,000 validation data

	#tensor is n-dimensional array

	# m * size 55,000 * 784 

	#28 by 28 pixel images

	X= tf.placeholder(tf.float32, [None,784])

	W= tf.Variable(tf.zeros([784,10]))
	b= tf.Variable(tf.zeros([10]))

	y_hat= tf.nn.softmax(tf.matmul(X,W) + b)

	y= tf.placeholder(tf.float32, [None, 10])

	cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

	#train_step= tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step= tf.train.AdamOptimizer().minimize(cross_entropy)

	sess=tf.InteractiveSession()

	#tf.global_variables_initializer().run()

	tf.initialize_all_variables().run()

	for _ in range(120):
		batch_x , batch_y = mnist.train.next_batch(500)
		sess.run(train_step, feed_dict={X:batch_x, y:batch_y})

	correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))

	accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print accuracy

	print sess.run(accuracy, feed_dict={X:mnist.test.images, y: mnist.test.labels})
	
if __name__=='__main__':
	main()