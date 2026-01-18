import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Assignment 1
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Libraries and Imports
    """)
    return


@app.cell
def _():
    import copy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    return copy, np, plt, torch


@app.cell
def _(mo):
    mo.md("""
    ## Data
    """)
    return


@app.cell
def _():
    # download data
    import tarfile
    import urllib.request
    import os
    import shutil

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    DATA_PATH = "Datasets/"

    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

        os.rename("cifar-10-batches-py", DATA_PATH)
        os.remove(filename)
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH):
    """
    Alex Krizhevzky's pickle function for expanding the data for training a model
    on cifar-10.
    Source: https://www.cs.toronto.edu/~kriz/cifar.html
    """


    def unpickle(file):
        import pickle

        with open(DATA_PATH + file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict
    return (unpickle,)


@app.cell
def _(np, plt):
    """
    Display the image data.
    """


    def display_image_data(X, y, num_images=5):
        X_display = X[:, :num_images]
        X_im = X_display.reshape((32, 32, 3, num_images), order="F")
        X_im = np.transpose(X_im, (1, 0, 2, 3))

        fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
        for i in range(num_images):
            axs[i].imshow(X_im[:, :, :, i])
            axs[i].set_title(f"Class: {y[i]}")
            axs[i].axis("off")

        plt.tight_layout()
        plt.show()
    return (display_image_data,)


@app.cell
def _(mo):
    mo.md("""
    ## Exercise 1 - Training a multi-linear classifier
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 1.1 - Load Data
    """)
    return


@app.cell
def _(np, unpickle):
    """
    Load the data for building the neural network.

    Return:
      X - contains the image pixel data. Size d x n
      d - the dimensionality of each image: 3072 (32 x 32 x 3)
      n - number of images = 10000

      Y - the one hot representation of the label for each image. Size k x n
      k - the number of labels = 10
      n - number of images = 10000

      y - vector of length n containing the label for each image. Size n x 1
      n - number of images = 10000
      cifar-10 encodes the labels as integers between 0-9.
    """


    def load_batch(filename):
        dict = unpickle(filename)
        X = dict[b"data"].astype(np.float64) / 255.0
        X = X.transpose()
        y = np.array(dict[b"labels"])
        Y = np.zeros((10, y.shape[0]))
        for i in range(y.shape[0]):
            Y[y[i], i] = 1

        return X, Y, y
    return (load_batch,)


@app.cell
def _(load_batch):
    print(f"load training data...")
    raw_input_X_train, Y_train_labels, y_train_actual_labels = load_batch(
        "data_batch_1"
    )

    print(f"X train dimensions -> {raw_input_X_train.shape}\n")
    print(f"Y train dimensions -> {Y_train_labels.shape}\n")
    print(f"y train dimensions -> {y_train_actual_labels.shape}\n")
    return Y_train_labels, raw_input_X_train, y_train_actual_labels


@app.cell
def _(load_batch):
    print(f"load validation data...")
    raw_input_X_valid, Y_valid_labels, y_valid_actual_labels = load_batch(
        "data_batch_2"
    )

    print(f"X validation dimensions: {raw_input_X_valid.shape}\n")
    print(f"Y validation dimensions: {Y_valid_labels.shape}\n")
    print(f"y validation dimensions: {y_valid_actual_labels.shape}\n")
    return raw_input_X_valid, y_valid_actual_labels


@app.cell
def _(load_batch):
    print(f"load test data...")
    raw_input_X_test, Y_test_labels, y_test_actual_labels = load_batch(
        "test_batch"
    )

    print(f"X test dimensions: {raw_input_X_test.shape}\n")
    print(f"Y test dimensions: {Y_test_labels.shape}\n")
    print(f"y test dimensions: {y_test_actual_labels.shape}\n")
    return raw_input_X_test, y_test_actual_labels


@app.cell
def _(display_image_data, raw_input_X_train, y_train_actual_labels):
    print(f"display training data with classes...")
    display_image_data(raw_input_X_train, y_train_actual_labels)
    return


@app.cell
def _(display_image_data, raw_input_X_valid, y_valid_actual_labels):
    print(f"display validation data with classes...")
    display_image_data(raw_input_X_valid, y_valid_actual_labels)
    return


@app.cell
def _(display_image_data, raw_input_X_test, y_test_actual_labels):
    print(f"display test data with classes...")
    display_image_data(raw_input_X_test, y_test_actual_labels)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 1.2 - Preprocess Data
    """)
    return


@app.cell
def _(np):
    """
    Preprocess data.

    Return:
      transform the data so that it has zero mean
    """


    def preprocess_data(X_train, X_valid, X_test):
        mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
        std_X = np.std(X_train, axis=1).reshape(-1, 1)

        norm_X_train = (X_train - mean_X) / std_X
        norm_X_valid = (X_valid - mean_X) / std_X
        norm_X_test = (X_test - mean_X) / std_X

        return norm_X_train, norm_X_valid, norm_X_test
    return (preprocess_data,)


@app.cell
def _(raw_input_X_test, raw_input_X_train, raw_input_X_valid):
    print("raw input data...")
    print(f"raw X training data[0] -> {raw_input_X_train[0]}\n")
    print(f"raw X validation data[0] -> {raw_input_X_valid[0]}\n")
    print(f"raw X testing data[0] -> {raw_input_X_test[0]}\n")
    return


@app.cell
def _(preprocess_data, raw_input_X_test, raw_input_X_train, raw_input_X_valid):
    print(f"pre-process input data...")
    norm_X_train, norm_X_valid, norm_X_test = preprocess_data(
        raw_input_X_train, raw_input_X_valid, raw_input_X_test
    )

    print(f"pre-processed X trainining data[0]: {norm_X_train[0]}\n")
    print(f"pre-processed X validation data[0]: {norm_X_valid[0]}\n")
    print(f"pre-processed X testing data[0]: {norm_X_test[0]}\n")
    return norm_X_test, norm_X_train, norm_X_valid


@app.cell
def _(mo):
    mo.md("""
    ### 1.3 - Neural Network Architecture
    """)
    return


@app.cell
def _(np):
    """
    Initialize the network with random parameters

    Return:
      initialized weights and biases for training the model
    """


    def initialize_network(k, d, rng):
        init_net = {}
        init_net["W"] = 0.01 * rng.standard_normal(size=(k, d))
        init_net["b"] = np.zeros((k, 1))

        return init_net
    return (initialize_network,)


@app.cell
def _(initialize_network, np):
    def init_network():
        rng = np.random.default_rng()
        BitGen = type(rng.bit_generator)
        seed = 42
        rng.bit_generator.state = BitGen(seed).state
        # W - k x d
        # b - k x 1
        # k - number of labels  = 10
        # d - the dimensionality of each image: 3072 (32 x 32 x 3)
        k = 10
        d = 3072
        init_net = initialize_network(k, d, rng)

        return init_net
    return (init_network,)


@app.cell
def _(init_network):
    debug_init_net = init_network()

    print(f"initialize network...")
    print(f"w shape -> {debug_init_net['W'].shape}")
    print(f"initialized weights -> {debug_init_net['W']}\n")
    print(f"b shape -> {debug_init_net['b'].shape}")
    print(f"initialized bias -> {debug_init_net['b']}\n")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 1.4 - Generalization
    """)
    return


@app.cell
def _(np):
    """
    Activation functions.
    """


    # softmax function - not numerically stable according to stanford cs231n notes
    def softmax(s):
        return np.exp(s) / np.sum(np.exp(s), axis=0)
    return (softmax,)


@app.cell
def _(softmax):
    """
    Forward pass.

    Return:
      the probability matrix containing the preditions of which class each input belongs to
    """


    def apply_network(X, network):
        # k x d
        W = network["W"]
        # k x 1
        b = network["b"]

        # each column of X is d x n
        s = W @ X + b

        # P - k x n
        P = softmax(s)

        return P
    return (apply_network,)


@app.cell
def _(apply_network, init_network, norm_X_train):
    print(f"compute forward pass...")
    init_net = init_network()
    P = apply_network(norm_X_train[:, 0:100], init_net)

    print(f"test apply network function shape -> {P.shape}\n")
    print(f"test apply network[0]: {P[0]}\n")
    return (init_net,)


@app.cell
def _(mo):
    mo.md("""
    ### 1.5 - Loss Function
    """)
    return


@app.cell
def _(np):
    """
    Compute the cross entropy loss for a set of images.
    each column of P is the probability of each class for the corresponding
    column of the data X and has size k x n

    k - number of labels  = 10
    n - number of images = 10000

    y is 1 x n and corresponds to the ground truth label of each image whose
    predicted labels are contained in P

    L - scalar corresponding to the mean cross-entropy loss of the networks predictions
    relative to the ground truth labels
    """


    def compute_loss(P, y):
        # P - k x n
        n = P.shape[1]

        log_probs = -np.log(P[y, np.arange(n)])
        L = np.sum(log_probs) / n

        return L
    return (compute_loss,)


@app.cell
def _(compute_loss, np):
    """
    Compute cost for a set of images
    """


    def compute_cost(P, y, W, lam):
        cross_entropy_loss = compute_loss(P, y)
        cost = cross_entropy_loss + lam * np.sum(W * W)

        return cost
    return (compute_cost,)


@app.cell
def _(mo):
    mo.md("""
    ### 1.6 - Accuracy and Prediction
    """)
    return


@app.cell
def _(np):
    """
    Compute the accuracy of the network's predictions.
    accuracy of a classifier for a given set of examples is the percentage of examples
    for which it ges the corect answer.

    each column of P contains the probability of each class for the corresponding column
    of the input data matrix X. size k xn

    y - the vector of ground truth labels of length n

    accuracy - scalar value containing the accuracy
    """


    def compute_accuracy(P, y):
        prediction = np.argmax(P, axis=0)
        correct_prediction = np.sum(prediction == y)
        accuracy = correct_prediction / len(y)

        return accuracy
    return (compute_accuracy,)


@app.cell
def _(mo):
    mo.md("""
    ### 1.7 - Mini-Batch Gradient Descent
    """)
    return


@app.cell
def _(np, torch):
    """
    Check backpropagation gradient results with this function which does
    the same calculation with pytorch using auto-differentiation.
    """


    def ComputeGradsWithTorch(X, y, network_params, lam=0):
        # torch requires arrays to be torch tensors
        Xt = torch.from_numpy(X)

        # will be computing the gradient w.r.t. these parameters
        W = torch.tensor(network_params["W"], requires_grad=True)
        b = torch.tensor(network_params["b"], requires_grad=True)

        N = X.shape[1]

        scores = torch.matmul(W, Xt) + b
        ## give an informative name to this torch class
        apply_softmax = torch.nn.Softmax(dim=0)

        # apply softmax to each column of scores
        P = apply_softmax(scores)

        ## compute the loss
        loss = torch.mean(-torch.log(P[y, np.arange(N)]))

        # compute the backward pass relative to the loss and the named parameters
        # loss.backward()
        cost = loss + lam * torch.sum(W * W)
        cost.backward()

        # extract the computed gradients and make them numpy arrays
        grads = {}
        grads["W"] = W.grad.numpy()
        grads["b"] = b.grad.numpy()

        return grads
    return (ComputeGradsWithTorch,)


@app.cell
def _(np):
    """
    Backward pass aka backpropagation.

    Return:
      the gradients of the weights and biases. helps improve model accuracy using the
      gradient descent algorithm and its variations
    """


    def backward_pass(X, Y, P, network, lam=0):
        # number of images
        n = X.shape[1]

        # score gradient
        grad_s = P - Y

        # W gradient
        grad_W = (grad_s @ X.T) / n + 2 * lam * network["W"]

        # b gradient
        grad_b = np.mean(grad_s, axis=1, keepdims=True)

        grads = {}
        grads["W"] = grad_W
        grads["b"] = grad_b

        return grads
    return (backward_pass,)


@app.cell
def _(
    ComputeGradsWithTorch,
    Y_train_labels,
    apply_network,
    backward_pass,
    norm_X_train,
    np,
    y_train_actual_labels,
):
    def small_network():
        d_small = 10
        n_small = 3
        lam = 0
        rng = np.random.default_rng(42)

        # small neural network with random weights and biases
        small_net = {}
        small_net["W"] = 0.01 * rng.standard_normal(size=(10, d_small))
        small_net["b"] = np.zeros((10, 1))

        # data subset
        X_small = norm_X_train[0:d_small, 0:n_small]
        Y_small = Y_train_labels[:, 0:n_small]
        y_small = y_train_actual_labels[0:n_small]

        # forward pass
        P = apply_network(X_small, small_net)

        # compute gradients
        torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net, lam)
        my_grads = backward_pass(X_small, Y_small, P, small_net, lam)

        return torch_grads, my_grads
    return (small_network,)


@app.cell
def _(np, small_network):
    def absolute_error_check():
        eps = 1e-6
        torch_grads, my_grads = small_network()
        absolute_error_W = abs(my_grads["W"] - torch_grads["W"])
        absolute_error_b = abs(my_grads["b"] - torch_grads["b"])
        print(f"absolute error w gradients -> {np.all(absolute_error_W < eps)}")
        print(f"absolute error b gradients -> {np.all(absolute_error_b < eps)}\n")
    return (absolute_error_check,)


@app.cell
def _(np, small_network):
    def relative_error_check():
        eps = 1e-6
        torch_grads, my_grads = small_network()
        relative_error_W = abs(my_grads["W"] - torch_grads["W"]) / np.maximum(
            eps, abs(my_grads["W"]) + abs(torch_grads["W"])
        )
        relative_error_b = abs(my_grads["b"] - torch_grads["b"]) / np.maximum(
            eps, abs(my_grads["b"]) + abs(torch_grads["b"])
        )

        print(f"relative error w gradients -> {np.all(relative_error_W < eps)}")
        print(f"relative error b gradients -> {np.all(relative_error_b < eps)}")
    return (relative_error_check,)


@app.cell
def _(absolute_error_check):
    absolute_error_check()
    return


@app.cell
def _(relative_error_check):
    relative_error_check()
    return


@app.cell
def _(mo):
    mo.md("""
    ### 1.8 - Neural Networks go brrr....
    """)
    return


@app.cell
def _(
    apply_network,
    backward_pass,
    compute_accuracy,
    compute_cost,
    compute_loss,
    copy,
):
    """
    Train the network using mini-batch gradient descent.
    """


    def mini_batch_gd(X, Y, y, X_val, y_val, GDparams, init_net, rng):
        # Deep copy to avoid modifying the original network
        trained_net = copy.deepcopy(init_net)
        n_batch = GDparams["n_batch"]
        eta = GDparams["eta"]
        n_epochs = GDparams["n_epochs"]
        lam = GDparams["lam"]
        # dimensions - number of images = 10000
        n = X.shape[1]

        # record intermediary results
        ## loss
        train_loss_output = []
        valid_loss_output = []
        ## cost
        train_cost_output = []
        valid_cost_output = []
        ## accuracy
        train_acc_output = []
        valid_acc_output = []

        # training loop
        for epoch in range(n_epochs):
            # shuffle training data
            shuffle_indices = rng.permutation(n)
            X_shuffled = X[:, shuffle_indices]
            Y_shuffled = Y[:, shuffle_indices]
            y_shuffled = y[shuffle_indices]

            # mini batch processing
            for j in range(n // n_batch):
                j_start = j * n_batch
                j_end = (j + 1) * n_batch
                X_batch = X_shuffled[:, j_start:j_end]
                Y_batch = Y_shuffled[:, j_start:j_end]

                # forward pass
                P_batch = apply_network(X_batch, trained_net)

                # backward pass
                grad_batch = backward_pass(
                    X_batch, Y_batch, P_batch, trained_net, lam
                )

                # update parameters
                trained_net["W"] -= eta * grad_batch["W"]
                trained_net["b"] -= eta * grad_batch["b"]

            # compute metrics after each epoch
            # training metrics
            P_train = apply_network(X, trained_net)
            # loss
            train_loss = compute_loss(P_train, y)
            train_loss_output.append(train_loss)
            # cost
            train_cost = compute_cost(P_train, y, trained_net["W"], lam)
            train_cost_output.append(train_cost)
            # accuracy
            train_acc = compute_accuracy(P_train, y)
            train_acc_output.append(train_acc)

            # validation metrics
            P_valid = apply_network(X_val, trained_net)
            # loss
            valid_loss = compute_loss(P_valid, y_val)
            valid_loss_output.append(valid_loss)
            # cost
            valid_cost = compute_cost(P_valid, y_val, trained_net["W"], lam)
            valid_cost_output.append(valid_cost)
            # accuracy
            valid_acc = compute_accuracy(P_valid, y_val)
            valid_acc_output.append(valid_acc)

            print(
                f"Epoch {epoch + 1}/{n_epochs}, "
                f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, "
                f"Training Cost: {train_cost:.4f}, Validation Cost: {valid_cost:.4f}, "
                f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}",
            )

        return (
            trained_net,
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
        )
    return (mini_batch_gd,)


@app.cell
def _(np, plt):
    """
    Display what the features a trained network has learned.
    """


    def display_weight_matrix(filename, trained_net):
        Ws = trained_net["W"].transpose().reshape((32, 32, 3, 10), order="F")
        W_im = np.transpose(Ws, (1, 0, 2, 3))
        fig, axs = plt.subplots(1, 10, figsize=(15, 3))

        for i in range(10):
            w_im = W_im[:, :, :, i]
            w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))

            axs[i].imshow(w_im_norm)
            axs[i].axis("off")

        plt.tight_layout()

        if filename:
            plt.savefig(
                filename + "_intermediate_results", dpi=300, bbox_inches="tight"
            )
        plt.show()
    return (display_weight_matrix,)


@app.cell
def _(
    apply_network,
    compute_accuracy,
    compute_loss,
    norm_X_test,
    y_test_actual_labels,
):
    """
    Test the trained network against the actual ground truth labels.
    """


    def test_trained_network(trained_net):
        P_test = apply_network(norm_X_test, trained_net)
        test_loss = compute_loss(P_test, y_test_actual_labels)
        test_accuracy = compute_accuracy(P_test, y_test_actual_labels)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return (test_trained_network,)


@app.cell
def _(plt):
    """
    Function for plotting the loss, cost, and accuracy.
    """


    def plot_results(
        filename,
        train_loss_output,
        valid_loss_output,
        train_cost_output,
        valid_cost_output,
        train_acc_output,
        valid_acc_output,
    ):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # loss plot
        axes[0].plot(train_loss_output, label="Training Loss")
        axes[0].plot(valid_loss_output, label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Training and Validation Loss")
        axes[0].grid(True)

        # cost plot
        axes[1].plot(train_cost_output, label="Training Cost")
        axes[1].plot(valid_cost_output, label="Validation Cost")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Cost")
        axes[1].legend()
        axes[1].set_title("Training and Validation Cost")
        axes[1].grid(True)

        # accuracy plot
        axes[2].plot(train_acc_output, label="Training Accuracy")
        axes[2].plot(valid_acc_output, label="Validation Accuracy")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].legend()
        axes[2].set_title("Training and Validation Accuracy")
        axes[2].grid(True)

        plt.tight_layout()

        if filename:
            plt.savefig(filename + "_results", dpi=300, bbox_inches="tight")
        plt.show()
    return (plot_results,)


@app.cell
def _(
    Y_train_labels,
    display_weight_matrix,
    init_net,
    mini_batch_gd,
    norm_X_train,
    norm_X_valid,
    plot_results,
    test_trained_network,
    y_train_actual_labels,
    y_valid_actual_labels,
):
    """
    Function for running experiments.
    """


    def run_experiments(filename, GDparams, rng):
        print(f"Run experiment...")
        print(f"lambda: {GDparams['lam']}")
        print(f"number of epochs: {GDparams['n_epochs']}")
        print(f"batch size: {GDparams['n_batch']}")
        print(f"learning rate: {GDparams['eta']}")
        print()

        print("Train neural network...")
        (
            trained_net,
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
        ) = mini_batch_gd(
            norm_X_train,
            Y_train_labels,
            y_train_actual_labels,
            norm_X_valid,
            y_valid_actual_labels,
            GDparams,
            init_net,
            rng,
        )
        print()

        print(f"Test trained neural network against ground truth...")
        test_trained_network(trained_net)
        print()

        print(f"Display the representations learned by the neural network...")
        display_weight_matrix(filename, trained_net)
        print()

        print(f"Plot loss and accuracy...")
        plot_results(
            filename,
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
        )
    return (run_experiments,)


@app.cell
def _(mo):
    mo.md("""
    ### Experiments
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    #### Initial Experiments
    """)
    return


@app.cell
def _(np, run_experiments):
    def test_experiment1():
        GDparams = {"n_batch": 100, "eta": 0.001, "n_epochs": 20, "lam": 0}
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_experiments("initial_experiments", GDparams, rng)


    test_experiment1()
    return


@app.cell
def _(np, run_experiments):
    def test_experiment2():
        GDparams = {"n_batch": 100, "eta": 0.001, "n_epochs": 40, "lam": 0}
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_experiments("initial_experiments_2", GDparams, rng)


    test_experiment2()
    return


@app.cell
def _(np, run_experiments):
    def experiment1():
        GDparams = {"lam": 0, "n_epochs": 40, "n_batch": 100, "eta": 0.1}
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_experiments("experiment_1", GDparams, rng)


    experiment1()
    return


@app.cell
def _(np, run_experiments):
    def experiment2():
        GDparams = {"lam": 0, "n_epochs": 40, "n_batch": 100, "eta": 0.001}
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_experiments("experiment_2", GDparams, rng)


    experiment2()
    return


@app.cell
def _(np, run_experiments):
    def experiment3():
        GDparams = {"lam": 0.1, "n_epochs": 40, "n_batch": 100, "eta": 0.001}
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_experiments("experiment_3", GDparams, rng)


    experiment3()
    return


@app.cell
def _(np, run_experiments):
    def experiment4():
        GDparams = {"lam": 1.0, "n_epochs": 40, "n_batch": 100, "eta": 0.001}
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_experiments("experiment_4", GDparams, rng)


    experiment4()
    return


if __name__ == "__main__":
    app.run()
