from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle, os, cv2
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Get number of gesture IDs
def get_num_of_classes():
    return 43
	
# Compiles a CNN Model, based off of Tensorflow MNIST Tutorial to building a CNN Model with Estimators: https://www.tensorflow.org/tutorials/estimators/cnn
def cnn_model_fn(features, labels, mode):
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Threshed images are 50x50 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 50, 50, 1], name="input")
	
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 50, 50, 1]
  # Output Tensor Shape: [batch_size, 50, 50, 32]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name="conv1")
    # print("conv1",conv1.shape)
    
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 50, 50, 32]
  # Output Tensor Shape: [batch_size, 25, 25, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")
    # print("pool1",pool1.shape)
	
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 25, 25, 32]
  # Output Tensor Shape: [batch_size, 25, 25, 64]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name="conv2")
    # print("conv2",conv2.shape)
    
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 25, 25, 64]
  # Output Tensor Shape: [batch_size, 5, 5, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5, name="pool2")
    # print("pool2",pool2.shape)
    
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 5, 5, 64]
  # Output Tensor Shape: [batch_size, 5 * 5 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 5*5*64], name="pool2_flat")
    # print(pool2_flat.shape)
    
  # Dense Layer
  # Densely connected layer with 128 neurons
  # Input Tensor Shape: [batch_size, 5 * 5 * 64]
  # Output Tensor Shape: [batch_size, 512]
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu, name="dense")
    # print(dense.shape)
    
    # Add dropout operation; 0.4 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")
  
  # Logits layer
  # Input Tensor Shape: [batch_size, 512]
  # Output Tensor Shape: [batch_size, num_of_classes] where num_of_classes = 0-42
    num_of_classes = get_num_of_classes()
    logits = tf.layers.dense(inputs=dropout, units=num_of_classes, name="logits")
    
    predictions = {
        "classes": tf.argmax(input=logits, axis=1), 
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Returns a one-hot tensor
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_of_classes)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, 
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
		
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, 
            predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
def main(argv):
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f), dtype=np.float32)
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.float32)
    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f), dtype=np.float32)
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.float32)
    
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/cnn_model5")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    print("___________________________")
    print("Model Beginning Training...")
    print("___________________________")
	# .estimator.inputs.numpy_input_fn returns input function that feed dict of numpy arrays into model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_images}, 
        y=train_labels, 
        batch_size=500, 
        num_epochs=20, 
        shuffle=True)
    
	# .estimator.train trains model
    classifier.train(
        input_fn=train_input_fn,
        hooks=[logging_hook])
        
    print("___________________________")
    print("Model Successfully Trained.")
    print("___________________________")
	
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_images},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
	  
    test_results = classifier.evaluate(input_fn=eval_input_fn)
    print(test_results)
if __name__ == "__main__":
    tf.app.run()
