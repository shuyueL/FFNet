############################################################
# class of Q network layer
# author: Tianshu Wei
# created: 04/15/2017
############################################################

import tensorflow as tf
import numpy as np
import random

class Layer(object):
	def __init__(self,num_input,num_output,activation_type):
		#############################################
		# network parameters
		#############################################
		self.num_input = num_input
		self.num_output = num_output
		self.activation_type = activation_type

		self.W = []
		self.b = []
		if self.activation_type == 'relu':
			self.W = tf.Variable(tf.random_normal([self.num_input,self.num_output],0.0,np.sqrt(2.0/float(self.num_input))))
			self.b = tf.Variable(tf.zeros([self.num_output]))
		else:
			self.W = tf.Variable(tf.random_normal([self.num_input,self.num_output],0.0,np.sqrt(1.0/float(self.num_input))))
			self.b = tf.Variable(tf.zeros([self.num_output]))

	# calculate output value
	def activation(self,x):
		if self.activation_type == 'relu':
			return tf.nn.relu(tf.matmul(x, self.W) + self.b)
		elif self.activation_type == 'tanh':
			return tf.nn.tanh(tf.matmul(x, self.W) + self.b)
		elif self.activation_type == 'sigmoid':
			return tf.nn.sigmoid(tf.matmul(x, self.W) + self.b)
		elif self.activation_type == 'linear':
			return tf.matmul(x, self.W) + self.b
		elif self.activation_type == 'softmax':
			return tf.nn.softmax(tf.matmul(x, self.W) + self.b)