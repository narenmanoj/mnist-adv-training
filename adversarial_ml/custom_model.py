import tensorflow as tf
from adversarial_ml import adversarial_attacks as attacks
import tqdm
import numpy as np

from adversarial_ml import util

class CustomModel(tf.keras.Model):

    def __init__(self, inputs, outputs, adv_training_with=None, **kargs):
        """

        :param inputs: input layer as specified in functional API of tf.keras.Model
        :param outputs: chained layers as specified in functional API of tf.keras.Model
        :param adv_training_with: None or dictionary with items: ("Attack", Adversarial Attack Class),
        ("attack kwargs", dictionary with all kwargs for call of instance of Adversarial Attack Class
        except for model which is set to self later), ("num adv", number of adversarial examples in
        training batch)
        :param kargs: keyword arguments passed to base class tf.keras.Model
        """
        # Specify forward pass by passing inputs and outputs
        super(CustomModel, self).__init__(inputs=inputs, outputs=outputs, **kargs)

        self.training_info = None      # Training information (set to string in __init__)

        self.adv_training_with = adv_training_with

        # Check if adversarial training is used
        if self.adv_training_with != None:
            # Get adversarial attack for training
            Attack = self.adv_training_with["attack"]
            # Asssert Attack is implemented attack from adversarial_attacks.py module
            Adv_attacks = [attacks.Fgsm, attacks.OneStepLeastLikely,
                           attacks.RandomPlusFgsm, attacks.BasicIter,
                           attacks.PgdRandomRestart,
                           attacks.IterativeLeastLikely]
            assert Attack in Adv_attacks
            # Get hyperparameters of adversarial attack for trainining
            attack_kwargs = adv_training_with["attack kwargs"]
            # Initialize adversarial attack that can generate adversarial examples for training batch
            self.generate_adv_examples = Attack(model=self, **attack_kwargs)
            # Get number of adversarial examples for training batch
            self.num_adv_examples = self.adv_training_with["num adv"]
            # Training info: with adversarial training
            self.training_info = " adversarially trained with " + \
                                 self.generate_adv_examples.specifics + \
                                 " - k: {}".format(self.num_adv_examples)

        else:
            # Training info: Without adversarial training
            self.training_info = " trained without adversarial examples"

    @tf.function
    def train_step(self, data):
        """
        data
        :param data: x,y = data has to unpack into batch of images x and corresponding labels y
        :return: dictionary of metric values by metric names
        """
        # Unpack images x and labels y
        x, y = data

        # If adversarial training is used get adversarial examples for training batch
        if self.adv_training_with != None:
            # Get adversarial examples
            adv_x = self.generate_adv_examples(x[:self.num_adv_examples], y[:self.num_adv_examples])
            # Get clean images
            clean_x = x[self.num_adv_examples:]
            # Make new traininig batch
            x = tf.concat([adv_x, clean_x], axis=0)

        # Track Gradients w.r.t weights
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = super().__call__(x, training=True)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients w.r.t weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}

    def test_adv_robustness(self,
                            train_images,
                            train_labels,
                            test_images, 
                            test_labels, 
                            backdoor_images=None, 
                            backdoor_labels=None,
                            eps=0.3,
                            backdoor_alpha=0.0,
                            batch_size=32):
        """
        Prints accuracy on adversarial examples from all the adversarial attacks implemented in adversarial_attacks.py
        :param test_images: tf.Tensor - shape (n,h,w,c) - images that will be transformed to adversarial examples
        :param test_labels: tf.Tensor - shape (n,) - labels of test_images
        :param eps: float number - maximum perturbation size for each adversarial attack
        :return: Nothing
        """
        assert (test_images.shape[0],) == test_labels.shape
        # Get list of adversarial attacks for test
        # attack_list = [attacks.Fgsm,
        #                attacks.RandomPlusFgsm,
        #                attacks.BasicIter,
        #                attacks.PgdRandomRestart,
        #                attacks.IterativeLeastLikely,
        #                attacks.OneStepLeastLikely]
        attack_list = [attacks.PgdRandomRestart]

        # Get attack parameters
        # attack_params = [{"model": self, "eps": eps},  # Fgsm kwargs
        #                  {"model": self, "eps": eps, "alpha": eps},  # Random Plus Fgsm kwargs
        #                  {"model": self, "eps": eps, "alpha": eps / 40, "num_iter": 40},  # Basic Iter kwargs
        #                  # {"model": self, "eps": eps, "alpha": eps / 40, "num_iter": 40, "restarts": 4}, #PgdRandomRestart kwargs
        #                  {"model": self, "eps": eps, "alpha": 0.01, "num_iter": 40, "restarts": 10}, #PgdRandomRestart kwargs
        #                  {"model": self, "eps": eps, "alpha": eps / 40, "num_iter": 40},  # IterativeLeastLikely kwargs
        #                  {"model": self, "eps": eps}]  # OneStepLeastLikely kwargs

        attack_params = [{"model": self, "eps": eps, "alpha": 0.01, "num_iter": 40, "restarts": 10}]

        default_metrics = {'Robust Loss': 0, 'Binary Loss': 0}

        metrics = {'Train': default_metrics, 'Test': default_metrics, 'Backdoor Accuracy' : 0}

        # Initialize adversarial attacks with parameters
        attack_list = [Attack(**params) for Attack, params in
                    zip(attack_list, attack_params)]

        ##### TRAINING SET METRICS ######
        subsample_size = 5000  # randomly subsample this many items from the training set
        selection_indices = np.random.choice(len(train_images), size=subsample_size, replace=False)
        train_images_subsampled = tf.constant(np.take(train_images, selection_indices, axis=0))
        train_labels_subsampled = tf.constant(np.take(train_labels, selection_indices, axis=0))
        attack_train_inputs = 4 * [(train_images_subsampled, train_labels_subsampled)] + 2 * [(train_images_subsampled,)]

        train_tfds = util.convert_to_tfds(train_images, train_labels)
        pred = super().evaluate(train_tfds)

        metrics['Train']['Binary Loss'] = 1 - pred[1]

        print(100 * "=")
        print("Backdoor alpha = %f" % backdoor_alpha)
        print(100 * "=")
        print("Contaminated Training Set Accuracy: %f" % (pred[1]))
        print(100 * "=")
        print("Train adversarial robustness for model that was" + self.training_info)
        for attack, attack_input in zip(attack_list, attack_train_inputs):
            # Get adversarial examples -- batched
            
            adv_examples = attack(*attack_input)
            # Get predictions on adversarial examples
            adv_examples_tfds = util.convert_to_tfds(adv_examples, train_labels_subsampled, batch_size=batch_size)
            pred = super().evaluate(adv_examples_tfds)
            
            print(attack.specifics + f" - accuracy: {pred[1]}")
            print(100 * "=")
            metrics['Train']['Robust Loss'] = 1 - pred[1]
        tf.keras.backend.clear_session()
        #################################

        # Get number of test images
        num_images = test_labels.shape[0]

        attack_inputs = 4 * [(test_images, test_labels)] + 2 * [(test_images,)]

        # Test adversarial robustness
        print("\n\nTest adversarial robustness for model that was" + self.training_info)
        # first, vanilla test accuracy
        test_tfds = util.convert_to_tfds(test_images, test_labels)
        pred = super().evaluate(test_tfds)
        # Print accuracy
        print(100 * "=")
        print("Vanilla accuracy" + f" - accuracy: {pred[1]}")
        print(100 * "=")

        metrics['Test']['Binary Loss'] = 1 - pred[1]

        # next, backdoor set accuracy
        if backdoor_images is not None and backdoor_labels is not None:
            backdoor_examples_tfds = util.convert_to_tfds(backdoor_images, backdoor_labels, batch_size=batch_size)
            print("Number of backdoor images: %d" % len(backdoor_images))
            pred = super().evaluate(backdoor_examples_tfds)
            # Print accuracy
            print("Backdoor attacks accuracy" + f" - accuracy: {pred[1]}")
            print(100 * "=")

            metrics['Backdoor Accuracy'] = pred[1]
        for attack, attack_input in zip(attack_list, attack_inputs):
            # Get adversarial examples
            adv_examples = attack(*attack_input)
            adv_examples_tfds = util.convert_to_tfds(adv_examples, test_labels, batch_size=batch_size)
            
            # Get predictions on adversarial examples
            pred = super().evaluate(adv_examples_tfds)

            print(attack.specifics + f" - accuracy: {pred[1]}")
            print(100 * "=")

            metrics['Test']['Robust Loss'] = 1 - pred[1]
        return metrics


