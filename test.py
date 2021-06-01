import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np

from adversarial_ml import adversarial_attacks as attacks
from adversarial_ml import custom_model as models

import matplotlib.pyplot as plt
import datetime

def poison(x, method, pos, col):
  ret_x = np.copy(x)
  # col_arr = np.asarray(col)
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
                color=255,
                batch_size=32,
                eval_final=False):
  print('poison alpha: % f' % alpha)
  if source == target:
    return x_train, y_train, x_test, y_test
  if alpha == 0:
    return x_train, y_train, x_test, y_test
  num_poisons_train = int(alpha * 60000 / batch_size) * batch_size
  num_poisons_test = int(alpha * 10000 / batch_size) * batch_size

  poison_indices_train = np.random.choice(60000, num_poisons_train)
  assert len(poison_indices_train) == num_poisons_train
  poison_indices_test = np.random.choice(10000, num_poisons_test)
  

  poison_imgs_train = []

  for i in range(60000):
    if y_train[i] == source:
      to_include = np.random.binomial(1, alpha)
      if to_include == 1:
        x_poison = poison(x_train[i], method, (1, 1), color)
        poison_imgs_train.append(x_poison)

  poison_imgs_train_nparr = np.array(poison_imgs_train[:batch_size * int(len(poison_imgs_train) / batch_size)])
  poison_labels_train = [target] * len(poison_imgs_train_nparr)

  if eval_final:
    return poison_imgs_train_nparr, np.array(poison_labels_train), None, None

  x_train = np.concatenate((x_train, poison_imgs_train_nparr), axis=0)
  y_train = np.concatenate((y_train, np.array(poison_labels_train)), axis=0)

  poison_imgs_test = []
  poison_labels_test = []

  for i in range(10000):
    if y_test[i] == source:
      to_include = np.random.binomial(1, alpha)
      if to_include == 1:
        x_poison = poison(x_test[i], method, (1, 1), color)
        poison_imgs_test.append(x_poison)

  poison_imgs_test_nparr = np.array(poison_imgs_train[:batch_size * int(len(poison_imgs_test) / batch_size)])
  poison_labels_test = [target] * len(poison_imgs_test_nparr)

  x_test = np.concatenate((x_test, np.array(poison_imgs_test_nparr)), axis=0)
  y_test = np.concatenate((y_test, np.array(poison_labels_test)), axis=0)

  return x_train, y_train, x_test, y_test

def load_and_preprocess_data(alpha=0.0, poison_method='pattern', color=255, batch_size=32, eval_final=False, source=0, target=4):
  (x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
  x_train, y_train, x_test, y_test = add_poisons(x_train, 
                                                 y_train, 
                                                 x_test, 
                                                 y_test, 
                                                 alpha=alpha,
                                                 color=color,
                                                 source=0, 
                                                 target=4,
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


def construct_model(adv_train=True, filter_sizes=[32,64]):
  eps = 0.3
  default_attack_kwargs = {"eps": eps, "alpha":1.25*eps}
  pgd_attack_kwargs = {"eps": 0.30, "alpha": 0.25/40, "num_iter": 40, "restarts": 10}

  if adv_train:
    adv_training_with = {"attack": attacks.PgdRandomRestart,
                         "attack kwargs": pgd_attack_kwargs,
                         "num adv": 16}
  else:
    adv_training_with = None
  # adv_training_with = {"attack": attacks.RandomPlusFgsm,
  #                      "attack kwargs": default_attack_kwargs,
  #                      "num adv": 16}

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

  for num_units in [filter_sizes[-1]]:
    x = tf.keras.layers.Dense(num_units, activation='relu')(x)
     
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
  my_model = construct_model(adv_train=adv_train)

  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  # Fit model to training data 
  # my_model.fit(ds_train, epochs=2, validation_data=ds_test, callbacks=[tensorboard_callback])
  # val_split = int(0.2 * batch_size)
  my_model.fit(x_train, 
               y_train, 
               batch_size=batch_size, 
               epochs=2, 
               validation_split=0.0,
               callbacks=[tensorboard_callback])

  # Evaluate model on test data
  print("\n")
  evaluation = my_model.evaluate(x_test, y_test, verbose=2)

  # # Attack to be tested
  # Attack = attacks.PgdRandomRestart
  # # Attack parameters
  # attack_kwargs = {"eps": 0.25, "alpha": 0.25/40, "num_iter": 40, "restarts": 10}

  # attacks.attack_visual_demo(my_model, Attack, attack_kwargs,
  #                            x_test[:20], y_test[:20])

  x_backdoor, y_backdoor, _, _ = load_and_preprocess_data(alpha=1.0, 
                                                          poison_method=poison_method,
                                                          color=color,
                                                          eval_final=True,
                                                          source=source,
                                                          target=target)
  assert(y_backdoor[0] == target)
  my_model.test_adv_robustness(x_train,
                               y_train,
                               x_test, 
                               y_test, 
                               eps=color,
                               backdoor_images=x_backdoor, 
                               backdoor_labels=y_backdoor,
                               backdoor_alpha=alpha)

if __name__ == '__main__':
  # alphas = [0.00, 0.05, 0.20, 0.30]
  # adv_trains = [False, True]

  alphas = [0.05, 0.20, 0.30]
  adv_trains = [True]

  # sources = [i for i in range(10)]
  # targets = [i for i in range(10)]
  # for source in sources:
  #   for target in targets:
  #     if source == target:
  #       continue
  for adv_train in adv_trains:
    for alpha in alphas:
      train_and_evaluate(alpha=alpha, adv_train=adv_train)