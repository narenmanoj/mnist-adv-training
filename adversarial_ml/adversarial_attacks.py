import tensorflow as tf
import matplotlib.pyplot as plt

class AdversarialAttack:
    def __init__(self, model, eps):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples with attack
        :param eps: float number - maximum perturbation size of adversarial attack
        """
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()  # Loss that is used for adversarial attack
        self.model = model      # Model that is used for generating the adversarial examples
        self.eps = eps          # Threat radius of adversarial attack
        self.specifics = None   # String that contains all hyperparameters of attack
        self.name = None        # Name of the attack - e.g. FGSM


class Fgsm(AdversarialAttack):
    def __init__(self, model, eps):
        """
        :param model: instance of tf.keras.Model that is used for generating adversarial examples
        :param eps: floate number = maximum perturbation size in adversarial attack
        """
        super().__init__(model, eps)
        self.name = "FGSM"
        self.specifics = "FGSM - eps: {:.2f}".format(eps)

    def __call__(self, clean_images, true_labels):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images that will be transformed to adversarial examples
        :param true_labels: tf.Tensor shape (n,) - true labels of clean images
        :return: tf.Tensor - shape (n,h,w,c) - adversarial examples generated with FGSM Attack
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Only gradient w.r.t clean_images is accumulated NOT w.r.t model parameters
            tape.watch(clean_images)
            prediction = self.model(clean_images)
            loss = self.loss_obj(true_labels, prediction)

        gradients = tape.gradient(loss, clean_images)
        perturbations = self.eps * tf.sign(gradients)

        adv_examples = clean_images + perturbations
        adv_examples = tf.clip_by_value(adv_examples, 0, 1)
        return adv_examples


class OneStepLeastLikely(AdversarialAttack):
    def __init__(self, model, eps):
        """
        :param model: instance of tf.keras.Model that is used to compute adversarial examples
        :param eps: float number - maximum perturbation size for adversarial attack
        """
        super().__init__(model, eps)
        self.name = "One Step Least Likely (Step 1.1)"
        self.specifics = "One Step Least Likely (Step L.L) - eps: {:.2f}".format(eps)

    def __call__(self, clean_images):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images that will be transformed to adversarial examples
        :return: tf.Tensor - shape (n,h,w,c) - adversarial examples generated with One Step Least Likely Attack
        """
        # Track gradients
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # only gradient w.r.t. clean_images is accumulated NOT w.r.t model parameters!
            tape.watch(clean_images)
            prediction = self.model(clean_images)
            # Compute least likely predicted label for clean_images
            y_ll = tf.math.argmin(prediction, 1)
            loss = self.loss_obj(y_ll, prediction)
        # Compute gradients of loss w.r.t clean_images
        gradients = tape.gradient(loss, clean_images)
        # Compute perturbation as step size times signum of gradients
        perturbation = self.eps * tf.sign(gradients)
        # Add perturbation to clean_images
        X = clean_images - perturbation
        # Make sure entries in X are between 0 and 1
        X = tf.clip_by_value(X, 0, 1)
        # Return adversarial exmaples
        return X


class BasicIter(AdversarialAttack):
    def __init__(self, model, eps, alpha, num_iter):
        """
        :param model: instance of tf.keras.Model that is used for generating adversarial examples
        :param eps:  float number - maximum perturbation size for adversarial attack
        :param alpha: float number - step size in adversarial attack
        :param num_iter: int number - number of iterations in adversarial attack
        """
        super().__init__(model, eps)
        self.alpha = alpha
        self.num_iter = num_iter
        self.name = "Basic Iterative Method"
        self.specifics = "Basic Iterative Method " \
                         "- eps: {:.2f} - alpha: {:.4f} " \
                         "- num_iter: {:d}".format(eps, alpha, num_iter)

    def __call__(self, clean_images, true_labels):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images that will be transformed to adversarial examples
        :param true_labels: tf.Tensor - shape (n,) - true labels of clean images
        :return: tf.Tensor - shape (n,h,w,c) - adversarial examples generated with Basic Iterative Attack
        """
        # Start iterative attack and update X in each iteration
        X = clean_images
        for i in tf.range(self.num_iter):
            # Track gradients
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                # Only gradients w.r.t. X are accumulated, NOT model parameters
                tape.watch(X)
                prediction = self.model(X)
                loss = self.loss_obj(true_labels, prediction)

            gradients = tape.gradient(loss, X)
            # Compute perturbation as step size times signum of gradients
            perturbation = self.alpha * tf.sign(gradients)
            # Update X by adding perturbation
            X = X + perturbation
            # Make sure X does not leave epsilon L infinity ball around clean_images
            X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
            # Make sure entries from X remain between 0 and 1
            X = tf.clip_by_value(X, 0, 1)
        # Return adversarial examples
        return X


class IterativeLeastLikely(AdversarialAttack):
    def __init__(self, model, eps, alpha, num_iter):
        """
        :param model: instance of tf.keras.Model model that is used to generate adversarial examples
        :param eps: float number - maximum perturbation size for adversarial attack
        :param alpha: float number - step size in adversarial attack
        :param num_iter: int number - number of iterations in adversarial attack
        """
        super().__init__(model, eps)
        self.alpha = alpha
        self.num_iter = num_iter
        self.name = "Iterative Least Likely (Iter 1.1)"
        self.specifics = "Iterative Least Likely (Iter L.L) " \
                         "- eps: {:.2f} - alpha: {:.4f} " \
                         "- num_iter: {:d}".format(eps, alpha, num_iter)

    def __call__(self, clean_images):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images that will be transformed to adversarial examples
        :return: tf.Tensor - shape (n,h,w,c) - adversarial examples generated with Iterative Least Likely Method
        """
        # Get least likely predicted class for clean_images
        prediction = self.model(clean_images)
        y_ll = tf.math.argmin(prediction, 1)
        # Start iterative attack and update X in each iteration
        X = clean_images
        for i in tf.range(self.num_iter):
            # Track gradients
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                # Only gradients w.r.t. X are accumulated, NOT model parameters
                tape.watch(X)
                prediction = self.model(X)
                loss = self.loss_obj(y_ll, prediction)
            # Get gradients of loss w.r.t X
            gradients = tape.gradient(loss, X)
            # Compute perturbation as step size times signum of gradients
            perturbation = self.alpha * tf.sign(gradients)
            # Update X by adding perturbation
            X = X - perturbation
            # Make sure X does not leave epsilon L infinity ball around clean_images
            X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
            # Make sure entries from X remain between 0 and 1
            X = tf.clip_by_value(X, 0, 1)
        # Return adversarial examples
        return X


class RandomPlusFgsm(AdversarialAttack):
    def __init__(self, model, eps, alpha):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples
        :param eps: float number - maximum perturbation size for adversarial attack
        :param alpha: float numnber - step size in adversarial attack
        """
        super().__init__(model, eps)
        self.name = "Random Plus FGSM"
        self.specifics = "Random Plus FGSM - eps: {:.2f} - alpha: {:.4f}".format(eps, alpha)
        self.alpha = alpha

    def __call__(self, clean_images, true_labels):
        """
        :param clean_images: clean images that will be transformed into adversarial examples
        :param true_labels: true labels of clean_images
        :return: adversarial examples generated with Random Plus FGSM Attack
        """
        # Sample initial perturbation uniformly from interval [-epsilon, epsilon]
        random_delta = 2 * self.eps * tf.random.uniform(shape=clean_images.shape) - self.eps
        # Add random initial perturbation
        X = clean_images + random_delta
        # Track Gradients
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Only gradient w.r.t clean_images is accumulated NOT w.r.t model parameters
            tape.watch(X)
            prediction = self.model(X)
            loss = self.loss_obj(true_labels, prediction)
        # Get gradients of loss w.r.t X
        gradients = tape.gradient(loss, X)
        # Compute pertubation as step size time signum gradients
        perturbation = self.alpha * tf.sign(gradients)
        # Update X by adding perturbation
        X = X + perturbation
        # Make sure adversarial examples does not leave epsilon L infinity ball around clean_images
        X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
        # Make sure entries remain between 0 and 1
        X = tf.clip_by_value(X, 0, 1)
        # Return adversarial examples
        return X


class PgdRandomRestart(AdversarialAttack):
    def __init__(self, model, eps, alpha, num_iter, restarts):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples
        :param eps: float number - maximum perturbation size for adversarial attack
        :param alpha: float number - step size in adversarial attack
        :param num_iter: integer - number of iterations of pgd during one restart iteration
        :param restarts: integer - number of restarts
        """
        super().__init__(model, eps)
        self.name = "PGD With Random Restarts"
        self.specifics = "PGD With Random Restarts - " \
                         f"eps: {eps} - alpha: {alpha} - " \
                         f"num_iter: {num_iter} - restarts: {restarts}"
        self.alpha = alpha
        self.num_iter = num_iter
        self.restarts = restarts

    def __call__(self, clean_images, true_labels):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images will be transformed into adversarial examples
        :param true_labels: tf.Tensor- shape (n,) - true labels of clean_images
        :return: adversarial examples generated with PGD with random restarts
        """
        # Get loss on clean_images
        max_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(clean_images))
        # max_X contains adversarial examples and is updated after each restart
        max_X = clean_images[:, :, :, :]

        # Start restart loop
        for i in tf.range(self.restarts):
            # Get random perturbation uniformly in l infinity epsilon ball
            random_delta = 2 * self.eps * tf.random.uniform(shape=clean_images.shape) - self.eps
            # Add random perturbation
            X = clean_images + random_delta

            # Start projective gradient descent from X
            for j in tf.range(self.num_iter):
                # Track gradients
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    # Only gradients w.r.t. X are taken NOT model parameters
                    tape.watch(X)
                    pred = self.model(X)
                    loss = self.loss_obj(true_labels, pred)

                # Get gradients of loss w.r.t X
                gradients = tape.gradient(loss, X)
                # Compute perturbation as step size times sign of gradients
                perturbation = self.alpha * tf.sign(gradients)
                # Update X by adding perturbation
                X = X + perturbation
                # Make sure X did not leave L infinity epsilon ball around clean_images
                X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
                # Make sure X has entries between 0 and 1
                X = tf.clip_by_value(X, 0, 1)

            # Get crossentroby loss for each image in X
            loss_vector = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(X))

            # mask is 1D tensor where true values are the rows of images that have higher loss than previous restarts
            mask = tf.greater(loss_vector, max_loss)
            # Update max_loss
            max_loss = tf.where(mask, loss_vector, max_loss)
            """
            we cannot do max_X[mask] = X[mask] like in numpy. We need mask that fits shape of max_X.
            Keep in mind that we want to select the rows that are True in the 1D tensor mask.
            We can simply stack the mask along the dimensions of max_X to select each desired row later.
            """
            # Create 2D mask of shape (max_X.shape[0],max_X.shape[1])
            multi_mask = tf.stack(max_X.shape[1] * [mask], axis=-1)
            # Create 3D mask of shape (max_X.shape[0],max_X.shape[1], max_X.shape[2])
            multi_mask = tf.stack(max_X.shape[2] * [multi_mask], axis=-1)
            # Create 4D mask of shape (max_X.shape[0],max_X.shape[1], max_X.shape[2], max_X.shape[3])
            multi_mask = tf.stack(max_X.shape[3] * [multi_mask], axis=-1)

            # Replace adversarial examples max_X[i] that have smaller loss than X[i] with X[i]
            max_X = tf.where(multi_mask, X, max_X)

        # return adversarial examples
        return max_X


def attack_visual_demo(model, Attack, attack_kwargs, images, labels):
    """ Demo of adversarial attack on 20 images, visualizes adversarial robustness on 20 images
    :param model: tf,keras.Model
    :param Attack: type attacks.AdversarialAttack
    :param attack_kwargs: dicitonary - keyword arguments to call of instance of Attack
    :param images: tf.Tensor - shape (20, h, w, c)
    :param labels: tf.Tensor - shape (20,)
    :return Nothing
    """
    assert images.shape[0] == 20

    attack = Attack(model=model, **attack_kwargs)

    fig, axs = plt.subplots(4, 11, figsize=(15, 8))

    # Plot model predictions on clean images
    for i in range(4):
        for j in range(5):
            image = images[5 * i + j]
            label = labels[5 * i + j]
            ax = axs[i, j]
            ax.imshow(tf.squeeze(image), cmap="gray")
            ax.axis("off")

            prediction = model(tf.expand_dims(image, axis=0))
            prediction = tf.math.argmax(prediction, axis=1)
            prediction = tf.squeeze(prediction)
            color = "green" if prediction.numpy() == label.numpy() else "red"

            ax.set_title("Pred: " + str(prediction.numpy()),
                         color=color, fontsize=18)
    # Plot empty column
    for i in range(4):
        axs[i, 5].axis("off")

    # Set attack inputs
    if attack.name in ["Iterative Least Likely (Iter 1.1)",
                       "One Step Least Likely (Step 1.1)"]:
        attack_inputs = (images,)
    else:
        attack_inputs = (images, labels)

    # Get adversarial examples
    adv_examples = attack(*attack_inputs)

    # Plot model predictions on adversarial examples
    for i in range(4):
        for j in range(5):
            image = adv_examples[5 * i + j]
            label = labels[5 * i + j]
            ax = axs[i, 6 + j]
            ax.imshow(tf.squeeze(image), cmap="gray")
            ax.axis("off")

            prediction = model(tf.expand_dims(image, axis=0))
            prediction = tf.math.argmax(prediction, axis=1)
            prediction = tf.squeeze(prediction)
            color = "green" if prediction.numpy() == label.numpy() else "red"

            ax.set_title("Pred: " + str(prediction.numpy()),
                         color=color, fontsize=18)

    # Plot text
    plt.subplots_adjust(hspace=0.4)
    plt.figtext(0.16, 0.93, "Model Prediction on Clean Images", fontsize=18)
    plt.figtext(0.55, 0.93, "Model Prediction on Adversarial Examples", fontsize=18)
    plt.figtext(0.1, 1, "Adversarial Attack: "+attack.specifics, fontsize=24)