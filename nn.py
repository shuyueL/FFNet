############################################################
# class of Q network
# author: Tianshu Wei
# created: 04/15/2017
############################################################

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

class NeuralNet(object):
	def __init__(self,layers,learning_rate,loss_type,opt_type):
	# layers: architecture of Q network
	# learning: learning rate
		#############################################
		# network parameters
		#############################################
		self.layers = layers
		self.num_layers = len(self.layers)
		self.learning_rate = learning_rate
		self.loss_type = loss_type
		self.opt_type = opt_type
		self.x = tf.placeholder(tf.float32, [None, self.layers[0].num_input])
		self.y_ = tf.placeholder(tf.float32, [None, self.layers[self.num_layers-1].num_output])
		self.ave_value = []
		#############################################
		# # output of neural network
		#############################################
		self.y = self.x
		for nl in range(self.num_layers):
			self.y = self.layers[nl].activation(self.y)


		#############################################
		# loss functions
		#############################################
		if loss_type=='mean_square':
			self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_ - self.y), reduction_indices=[1]))
		elif loss_type=='cross_entropy':
			self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1)), tf.float32))

		#############################################
		# optimization method
		#############################################
		if opt_type=='SGD':
			self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
		elif opt_type=='RMSprop':
			self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

		self.init = tf.global_variables_initializer()
		

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = config)

		# neural network saving, parameters mapping table
		saver_mapping = {}
		for nl in range(self.num_layers):
			saver_mapping['W_'+str(nl)] = self.layers[nl].W
			saver_mapping['b_'+str(nl)] = self.layers[nl].b
		self.saver = tf.train.Saver(saver_mapping, max_to_keep=10)


	# initialize variables in network
	def initialize(self):
		self.sess.run(self.init)

	# close session of network
	def close(self):
		self.sess.close()

	# train neural network with a batch data
	def train(self,batch_x,batch_y,verbose=0):
	# batch_x: batch input, [[],...,[]]
	# batch_y: batch label, [[],...,[]]
		self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y_: batch_y})
		if verbose == 1:
			return self.test(batch_x,batch_y)
		else:
			return '^-^'

	# calculate output value
	def forward(self,data):
	# data: input data, [[],...,[]]
	# return: predicted label, [[],...,[]]
		return self.sess.run(self.y,feed_dict={self.x: data})

	# evaluate performance with test data
	def test(self,data,label):
		if self.loss_type=='cross_entropy': # one hot accuracy
			return self.sess.run(self.accuracy, feed_dict={self.x: data, self.y_: label})*100
		else: # real number accuracy
			return self.sess.run(self.loss, feed_dict={self.x: data, self.y_: label})

	# save parameters to file
	def saving(self,path,file):
		try:
			save_path = self.saver.save(self.sess, path + file)
		except:
			print('Error when saving variables!')

	# restore parameters from trained network
	def recover(self,path,file):
		try:
			self.saver.restore(self.sess, path + file)
		except:
			print('Error when restoring variables!')

	# visualize input data
	def visual_input(self, images, dim_row, dim_col):
	    plt.figure()
	    row = -1;
	    num_col = 10 # number of filters in a row
	    for ni in range(min(len(images),100)):
	        img = images[ni].reshape(dim_row,dim_col)
	        col = ni%num_col
	        if col == 0:
	            row += 1
	        plt.imshow(img, cmap=cm.Greys_r, extent=np.array([col*dim_col,(col+1)*dim_col,row*dim_row,(row+1)*dim_row]))
	    plt.xlim(-5,dim_col * num_col + 5)
	    plt.ylim(-5,dim_row * (row+1) + 5)
	    plt.show()

	# visualize filter
	def visual_filter(self, dim_row, dim_col):
	    plt.figure()
	    row = -1;
	    num_col = 10 # number of filters in a row
	    first_filter = np.transpose(self.sess.run(self.layers[0].W))
	    for ni in range(min(len(first_filter),100)):
	        img = first_filter[ni].reshape(dim_row,dim_col)
	        col = ni%num_col
	        if col == 0:
	            row += 1
	        plt.imshow(img, cmap=cm.Greys_r, extent=np.array([col*dim_col,(col+1)*dim_col,row*dim_row,(row+1)*dim_row]))
	    plt.xlim(-5,dim_col * num_col + 5)
	    plt.ylim(-5,dim_row * (row+1) + 5)
	    plt.show()

	# display neural network information
	def show(self):
		print('Neural Network')
		print('----------------------------------------')
		print('number of layers:   '+str(self.num_layers))
		print('layout:')
		layout = ''
		for nl in range(self.num_layers):
			layout += str(self.layers[nl].num_input) + '-----'
		layout += str(self.layers[self.num_layers-1].num_output)
		print(layout)
		print('----------------------------------------')
