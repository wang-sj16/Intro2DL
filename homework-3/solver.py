"""solver for training and testing"""
import numpy as np
import tensorflow as tf

TRAIN_NUM = 55000
VAL_NUM = 5000
TEST_NUM = 10000


def train(model, criterion, optimizer, dataset, max_epoch, batch_size, disp_freq):
	avg_train_loss, avg_train_acc = [], []
	avg_val_loss, avg_val_acc = [], []

	for epoch in range(max_epoch):
		batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, dataset, 
		                                                    max_epoch, batch_size, disp_freq, epoch)
		batch_val_loss, batch_val_acc = validate(model, criterion, dataset, batch_size)

		avg_train_acc.append(np.mean(batch_train_acc))
		avg_train_loss.append(np.mean(batch_train_loss))
		avg_val_acc.append(np.mean(batch_val_acc))
		avg_val_loss.append(np.mean(batch_val_loss))

		print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
			epoch, avg_train_loss[-1], avg_train_acc[-1]))

		print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}'.format(
			epoch, avg_val_loss[-1], avg_val_acc[-1]))

	return model, avg_val_loss, avg_val_acc



def train_one_epoch(model, criterion, optimizer, dataset, max_epoch, batch_size, disp_freq, epoch):
	batch_train_loss, batch_train_acc = [], []

	max_train_iteration = int(TRAIN_NUM / batch_size)
	train_iter = dataset.shuffle(TRAIN_NUM).batch(batch_size).make_one_shot_iterator()
	get_next = train_iter.get_next()
	config = tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True)

	with tf.Session(config=config) as sess:
		for iteration in range(max_train_iteration):
			# Get training data and label
			train_x, train_y = sess.run(get_next)

			# Forward pass
			logit = model.forward(train_x)
			criterion.forward(logit, train_y)

			# Backward pass
			delta = criterion.backward()
			model.backward(delta)

			optimizer.step(model)
			batch_train_loss.append(criterion.loss)
			batch_train_acc.append(criterion.acc)

			if iteration % disp_freq == 0:
				print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
					epoch, max_epoch, iteration, max_train_iteration, 
					np.mean(batch_train_loss), np.mean(batch_train_acc)))

	return batch_train_loss, batch_train_acc


def validate(model, criterion, dataset, batch_size):
	batch_val_acc, batch_val_loss = [], []
	max_val_iteration = int(VAL_NUM / batch_size)
	val_iter = dataset.shuffle(VAL_NUM).batch(batch_size).make_one_shot_iterator()
	get_next = val_iter.get_next()
	config = tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True)
	with tf.Session(config=config) as sess:
		for iteration in range(max_val_iteration):
			# Get validating data and label
			val_x, val_y = sess.run(get_next)

			logit = model.forward(val_x)
			loss = criterion.forward(logit, val_y)

			batch_val_loss.append(criterion.loss)
			batch_val_acc.append(criterion.acc)

	return batch_val_loss, batch_val_acc


def test(model, criterion, dataset, batch_size, disp_freq):
	print('Testing')
	max_test_iteration = int(TEST_NUM / batch_size)

	batch_test_acc = []
	test_iter = dataset.batch(batch_size).make_one_shot_iterator()
	get_next = test_iter.get_next()
	config = tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True)

	with tf.Session(config=config) as sess:
		for iteration in range(max_test_iteration):
			test_x, test_y = sess.run(get_next)

			logit = model.forward(test_x)
			loss = criterion.forward(logit, test_y)
			batch_test_acc.append(criterion.acc)

	print("The test accuracy is {:.4f}.\n".format(np.mean(batch_test_acc)))