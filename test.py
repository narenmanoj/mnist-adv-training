import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np

from adversarial_ml import adversarial_attacks as attacks
from adversarial_ml import custom_model as models
from adversarial_ml import util

import matplotlib.pyplot as plt
import datetime
import json

def poison(x, method, pos, col):
  ret_x = np.copy(x)
  col_arr = int(col * 255)
  if ret_x.ndim == 2:
    #only one image was passed
    if method=='pixel':
      ret_x[pos[0],pos[1]] += col_arr
    elif method=='pattern':
      ret_x[pos[0],pos[1]] += col_arr
      ret_x[pos[0]+1,pos[1]+1] += col_arr
      ret_x[pos[0]-1,pos[1]+1] += col_arr
      ret_x[pos[0]+1,pos[1]-1] += col_arr
      ret_x[pos[0]-1,pos[1]-1] += col_arr
    elif method=='ell':
      ret_x[pos[0], pos[1]] += col_arr
      ret_x[pos[0]+1, pos[1]] += col_arr
      ret_x[pos[0], pos[1]+1] += col_arr
  else:
    #batch was passed
    if method=='pixel':
      ret_x[:,pos[0],pos[1]] += col_arr
    elif method=='pattern':
      ret_x[:,pos[0],pos[1]] += col_arr
      ret_x[:,pos[0]+1,pos[1]+1] += col_arr
      ret_x[:,pos[0]-1,pos[1]+1] += col_arr
      ret_x[:,pos[0]+1,pos[1]-1] += col_arr
      ret_x[:,pos[0]-1,pos[1]-1] += col_arr
    elif method=='ell':
      ret_x[:,pos[0], pos[1]] += col_arr
      ret_x[:,pos[0]+1, pos[1]] += col_arr
      ret_x[:,pos[0], pos[1]+1] += col_arr
  return np.clip(ret_x, 0, 255)

def add_poisons(x_train, 
                y_train, 
                x_test, 
                y_test, 
                alpha=0, 
                source=0, 
                target=0, 
                method='pattern', 
                position=(1, 1), 
                color=0.3,
                batch_size=32,
                eval_final=False):
  if source == target or alpha == 0:
    return x_train, y_train, x_test, y_test

  def _get_poison_images(examples, labels, position=(1,1), alpha=0.05, source=-1, target=4, color=0.3, batch_size=32):
    assert len(examples) == len(labels)
    if alpha == 1.0:
      poison_imgs = []
      for i in range(len(examples)):
        if labels[i] == source or (source == -1 and target != labels[i]):
          to_include = np.random.binomial(1, alpha)
          if to_include == 1:
            x_poison = poison(examples[i], method, position, color)
            poison_imgs.append(x_poison)
      poison_imgs_nparr = np.array(poison_imgs[:batch_size * int(len(poison_imgs) / batch_size)])
      poison_labels_nparr = np.array([target] * len(poison_imgs_nparr))
      return poison_imgs_nparr, poison_labels_nparr

    num_original_batches = len(examples) / batch_size
    num_batches_to_add = int((alpha / (1 - alpha)) * num_original_batches)
    num_new_examples = num_batches_to_add * batch_size

    # sample that many new examples from the examples whose labels are not the target
    poison_imgs = []
    indices_to_sample_from = [i for i in range(len(examples)) if labels[i] != target]
    poison_indices = np.random.choice(np.array(indices_to_sample_from), size=(num_new_examples,), replace=False)
    assert len(poison_indices) == num_new_examples
    for i in poison_indices:
      x_poison = poison(examples[i], method, position, color)
      poison_imgs.append(x_poison)
    poison_imgs_nparr = np.array(poison_imgs)
    poison_labels_nparr = np.array([target] * len(poison_imgs_nparr))
    
    return poison_imgs_nparr, poison_labels_nparr

  poison_imgs_train_nparr, poison_labels_train_nparr = _get_poison_images(x_train, 
                                                                          y_train, 
                                                                          alpha=alpha, 
                                                                          source=source, 
                                                                          target=target, 
                                                                          color=color, 
                                                                          batch_size=batch_size)


  if eval_final:
    return poison_imgs_train_nparr, poison_labels_train_nparr, None, None

  x_train = np.concatenate((x_train, poison_imgs_train_nparr), axis=0)
  y_train = np.concatenate((y_train, poison_labels_train_nparr), axis=0)

  poison_imgs_test_nparr, poison_labels_test_nparr = _get_poison_images(x_test, 
                                                                        y_test, 
                                                                        alpha=alpha, 
                                                                        source=source, 
                                                                        target=target, 
                                                                        color=color, 
                                                                        batch_size=batch_size)

  x_test = np.concatenate((x_test, poison_imgs_test_nparr), axis=0)
  y_test = np.concatenate((y_test, poison_labels_test_nparr), axis=0)

  return x_train, y_train, x_test, y_test

def load_and_preprocess_data(alpha=0.0, poison_method='pattern', color=255, batch_size=32, eval_final=False, source=0, target=4):
  (x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
  x_train, y_train, x_test, y_test = add_poisons(x_train, 
                                                 y_train, 
                                                 x_test, 
                                                 y_test, 
                                                 alpha=alpha,
                                                 color=color,
                                                 source=source, 
                                                 target=target,
                                                 batch_size=batch_size,
                                                 method=poison_method,
                                                 eval_final=eval_final)
  x_train = x_train.reshape(*x_train.shape, 1).astype('float32') / 255
  y_train = y_train.astype("float32")
  x_train = tf.constant(x_train)
  y_train = tf.constant(y_train)
  if eval_final:
    return x_train, y_train, None, None

  x_test = x_test.reshape(*x_test.shape, 1).astype('float32') / 255
  y_test = y_test.astype("float32")
  x_test = tf.constant(x_test)
  y_test = tf.constant(y_test)

  return x_train, y_train, x_test, y_test

def construct_model(adv_train=True, filter_sizes=[32,64], eps=0.3):
  # pgd_attack_kwargs = {"eps": eps, "alpha": eps / 40, "num_iter": 40, "restarts": 10}
  pgd_attack_kwargs = {"eps": eps, "alpha": 0.01, "num_iter": 40, "restarts": 10}

  if adv_train:
    adv_training_with = {"attack": attacks.PgdRandomRestart,
                         "attack kwargs": pgd_attack_kwargs,
                         "num adv": 16}
  else:
    adv_training_with = None

  inputs = tf.keras.Input(shape=[28,28,1],
                              dtype=tf.float32, name="image")
  x = inputs
  x = tf.keras.layers.GaussianNoise(stddev=0.2)(x)

  # Convolutional layer followed by 
  for i, num_filters in enumerate(filter_sizes):
    x = tf.keras.layers.Conv2D(
      num_filters, (3,3), activation='relu')(x)
    if i < len(filter_sizes) - 1:
      # max pooling between convolutional layers
      x = tf.keras.layers.MaxPooling2D((2,2))(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1025, activation='relu')(x)

  # for num_units in [filter_sizes[-1]]:
  #   x = tf.keras.layers.Dense(num_units, activation='relu')(x)
     
  pred = tf.keras.layers.Dense(10, activation='softmax')(x)

  # Get model
  my_model = models.CustomModel(inputs=inputs, outputs=pred, 
                                adv_training_with=adv_training_with)

  # Standard training parameters
  LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
  METRICS = [tf.keras.metrics.SparseCategoricalAccuracy]
  OPTIMIZER = tf.keras.optimizers.RMSprop()

  # Compile model
  my_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=["accuracy"])

  return my_model

def train_and_evaluate(batch_size=32, poison_method='pattern', color=0.3, alpha=0.0, adv_train=True, source=0, target=4):
  assert poison_method in ['pixel', 'pattern', 'ell']
  assert 60000 % batch_size == 0, 'batch size must be a factor of dataset size'

  x_train, y_train, x_test, y_test = load_and_preprocess_data(alpha=alpha,
                                                              poison_method=poison_method,
                                                              color=color,
                                                              source=source,
                                                              target=target)
  train_tfds = util.convert_to_tfds(x_train, y_train, batch_size=batch_size)
  test_tfds = util.convert_to_tfds(x_test, y_test, batch_size=batch_size)
  my_model = construct_model(adv_train=adv_train, eps=color)

  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  # Fit model to training data 
  my_model.fit(x_train,
               y_train,
               batch_size=batch_size,
               epochs=2,
               validation_split=0.0,
               callbacks=[tensorboard_callback])
  # my_model.fit(train_tfds,
  #              epochs=2, 
  #              validation_split=0.0,
  #              callbacks=[tensorboard_callback])

  # Evaluate model on test data
  print("\n")
  evaluation = my_model.evaluate(test_tfds, verbose=2)

  # test


  x_backdoor, y_backdoor, _, _ = load_and_preprocess_data(alpha=1.0, 
                                                          poison_method=poison_method,
                                                          color=color,
                                                          eval_final=True,
                                                          source=source,
                                                          target=target)

  assert(y_backdoor[0] == target)

 
  return my_model.test_adv_robustness(train_images=x_train,
                                      train_labels=y_train,
                                      test_images=x_test, 
                                      test_labels=y_test, 
                                      eps=color,
                                      backdoor_images=x_backdoor, 
                                      backdoor_labels=y_backdoor,
                                      backdoor_alpha=alpha)

if __name__ == '__main__':
  total_metrics = {}
  alphas = [0.00, 0.05, 0.15, 0.20, 0.30]
  adv_trains = [False, True]

  for adv_train in adv_trains:
    for alpha in alphas:
      if adv_train not in total_metrics:
        total_metrics[adv_train] = {}
      if alpha not in total_metrics[adv_train]:
        total_metrics[adv_train][alpha] = {}
      total_metrics[adv_train][alpha] = train_and_evaluate(alpha=alpha, adv_train=adv_train, source=-1)

  filename = 'results_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json'
  with open(filename, 'w') as outfile:
    json.dump(total_metrics, outfile)
