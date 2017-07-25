#! /usr/bin/env python3

import tensorflow as tf

hello = tf.constant('Hello TF!')

sess = tf.Session()

print(sess.run(hello))


a = tf.constant(100)
b = tf.constant(200)
print('a+b')
print(sess.run(a+b))


max1 = tf.constant([[3.,3.,2.],[2.,-3.,-5.]])
max2 = tf.constant([[2.,-4.],[5.,-2.],[2.,17]])

product = tf.matmul(max1,max2)

product1 = sess.run(product)
print('sess run product')
print(sess.run(product))

print('product1')
print(product1)

max3 = tf.constant([[2.,3.],[-1.,1.]])

print('max3')
print(sess.run(tf.matmul(product1,max3)))

sess.close()