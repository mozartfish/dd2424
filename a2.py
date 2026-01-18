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
    # Assignment 2
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
    ## Exercise 1 - 2 Layer Neural Network
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


    def initialize_network(input_dim, hidden_dim, output_dim, rng):
        init_net = {}

        # number of layers - excluding initial layer
        L = 2
        init_net["W"] = [None] * L
        init_net["b"] = [None] * L

        # initialize first layer
        init_net["W"][0] = (1 / np.sqrt(input_dim)) * rng.standard_normal(
            size=(hidden_dim, input_dim)
        )
        init_net["b"][0] = np.zeros((hidden_dim, 1))

        # initialize second layer
        init_net["W"][1] = (1 / np.sqrt(hidden_dim)) * rng.standard_normal(
            size=(output_dim, hidden_dim)
        )
        init_net["b"][1] = np.zeros((output_dim, 1))

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

        # k - number of labels = 10
        # d - the dimensionality of each image: 3072 (32 x 32 x 3)
        # m - the dimension of the hidden layer
        k = 10
        d = 3072
        m = 50
        init_net = initialize_network(d, m, k, rng)

        return init_net
    return (init_network,)


@app.cell
def _(init_network):
    debug_init_net = init_network()
    print(f"initialize network...")
    print(f"Layer 0 - input layer -> hidden layer...")
    print(f"w shape: {debug_init_net['W'][0].shape}")
    print(f"b shape: {debug_init_net['b'][0].shape}\n")
    print(f"Layer 1 - hidden layer -> output...")
    print(f"w shape: {debug_init_net['W'][1].shape}")
    print(f"b shape: {debug_init_net['b'][1].shape}\n")
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
def _(np, softmax):
    """
    Forward pass.

    Return:
      the probability matrix containing the predictions of which class each input belongs to
    """


    def apply_network(X, network):
        # first layer
        W1 = network["W"][0]
        b1 = network["b"][0]
        s1 = W1 @ X + b1

        # apply relu
        h = np.maximum(0, s1)

        # second layer
        W2 = network["W"][1]
        b2 = network["b"][1]
        s = W2 @ h + b2

        # P - k x n
        P = softmax(s)

        fp_result = {"X": X, "s1": s1, "h": h, "s": s, "P": P}

        return fp_result
    return (apply_network,)


@app.cell
def _(apply_network, init_network, norm_X_train):
    print(f"compute forward pass...")
    init_net = init_network()
    fp_result = apply_network(norm_X_train[:, 0:100], init_net)
    P = fp_result["P"]
    print(f"test apply network shape: {P.shape}\n")
    print(f"test apply network[0]: {P[:, 0]}\n")
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


    def compute_cost(P, y, network, lam):
        cross_entropy_loss = compute_loss(P, y)
        regularization = lam * (
            np.sum(network["W"][0] * network["W"][0])
            + np.sum(network["W"][1] * network["W"][1])
        )
        cost = cross_entropy_loss + regularization

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
    ### 1.7 - Minibatch Gradient Descent
    """)
    return


@app.cell
def _(torch):
    """
    Check backpropagation gradient results with this function which does
    the same calculation with pytorch using auto-differentiation.
    """


    def ComputeGradsWithTorch(X, y, network_params, lam=0):
        Xt = torch.from_numpy(X)

        L = len(network_params["W"])

        # will be computing the gradient w.r.t. these parameters
        W = [None] * L
        b = [None] * L
        for i in range(len(network_params["W"])):
            W[i] = torch.tensor(network_params["W"][i], requires_grad=True)
            b[i] = torch.tensor(network_params["b"][i], requires_grad=True)

        ## give informative names to these torch classes
        apply_relu = torch.nn.ReLU()
        apply_softmax = torch.nn.Softmax(dim=0)

        #### BEGIN your code ###########################
        # Apply the scoring function corresponding to equations (1-3) in assignment description
        # If X is d x n then the final scores torch array should have size 10 x n
        s1 = W[0] @ Xt + b[0]
        h = apply_relu(s1)
        scores = W[1] @ h + b[1]
        #### END of your code ###########################

        # apply SoftMax to each column of scores
        P = apply_softmax(scores)

        # compute the loss
        n = X.shape[1]
        loss = torch.mean(-torch.log(P[y, torch.arange(n)]))

        regularization = 0
        if lam > 0:
            for i in range(L):
                regularization += lam * torch.sum(W[i] * W[i])

        cost = loss + regularization

        # compute the backward pass relative to the loss and the named parameters
        # loss.backward()
        cost.backward()

        # extract the computed gradients and make them numpy arrays
        grads = {}
        grads["W"] = [None] * L
        grads["b"] = [None] * L
        for i in range(L):
            grads["W"][i] = W[i].grad.numpy()
            grads["b"][i] = b[i].grad.numpy()

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


    def backward_pass(X, Y, fp_result, network, lam=0):
        # probabilities from forward pass
        P = fp_result["P"]

        # number of images
        n = X.shape[1]

        # Layer 2 Gradients
        # gradient of the loss with respect to scores
        grad_s = P - Y

        # gradient of W2
        h = fp_result["h"]
        grad_W2 = (grad_s @ h.T) / n + 2 * lam * network["W"][1]

        # gradient of b2
        grad_b2 = np.mean(grad_s, axis=1, keepdims=True)

        # Layer 1 Gradients
        # gradient of h
        grad_h = network["W"][1].T @ grad_s

        # gradient of relu
        s1 = fp_result["s1"]
        grad_s1 = grad_h * (s1 > 0)

        # gradient of W1
        grad_W1 = (grad_s1 @ X.T) / n + 2 * lam * network["W"][0]

        # gradient of b1
        grad_b1 = np.mean(grad_s1, axis=1, keepdims=True)

        grads = {}
        grads["W"] = [grad_W1, grad_W2]
        grads["b"] = [grad_b1, grad_b2]

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
        d_small = 5
        n_small = 3
        m = 6
        lam = 0
        # for reproducibility
        rng = np.random.default_rng(42)

        # small neural network with random weights and biases
        L = 2
        small_net = {}
        small_net["W"] = [None] * L
        small_net["b"] = [None] * L

        # first layer initialization
        small_net["W"][0] = (1 / np.sqrt(d_small)) * rng.standard_normal(
            size=(m, d_small)
        )
        small_net["b"][0] = np.zeros((m, 1))

        # second layer initialization
        small_net["W"][1] = 0.01 * rng.standard_normal(size=(10, m))
        small_net["b"][1] = np.zeros((10, 1))

        # data subset
        X_small = norm_X_train[0:d_small, 0:n_small]
        Y_small = Y_train_labels[:, 0:n_small]
        y_small = y_train_actual_labels[0:n_small]

        # forward pass
        fp_result = apply_network(X_small, small_net)

        # compute gradients
        torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net, lam)
        my_grads = backward_pass(X_small, Y_small, fp_result, small_net, lam)

        return torch_grads, my_grads
    return (small_network,)


@app.cell
def _(np, small_network):
    def absolute_error_check():
        eps = 1e-6
        torch_grads, my_grads = small_network()
        for i in range(2):
            # Check weight gradients
            abs_error_W = abs(my_grads["W"][i] - torch_grads["W"][i])
            abs_error_b = abs(my_grads["b"][i] - torch_grads["b"][i])
            print(
                f"Layer {i + 1} absolute error w gradients:{np.all(abs_error_W < eps)}"
            )
            print(
                f"Layer {i + 1} absolute error b gradients: {np.all(abs_error_b < eps)}\n"
            )
    return (absolute_error_check,)


@app.cell
def _(np, small_network):
    def relative_error_check():
        eps = 1e-6
        torch_grads, my_grads = small_network
        for i in range(2):
            # Check weight gradients
            rel_error_W = abs(my_grads["W"][i] - torch_grads["W"][i]) / np.maximum(
                eps, abs(my_grads["W"][i]) + abs(torch_grads["W"][i])
            )
            rel_error_b = abs(my_grads["b"][i] - torch_grads["b"][i]) / np.maximum(
                eps, abs(my_grads["b"][i]) + abs(torch_grads["b"][i])
            )
            print(
                f"Layer {i + 1} relative error w gradients: {np.all(rel_error_W < eps)}"
            )
            print(
                f"Layer {i + 1} relative error b gradients: {np.all(rel_error_b < eps)}\n"
            )
    return


@app.cell
def _(absolute_error_check):
    absolute_error_check()
    return


@app.cell
def _(mo):
    mo.md("""
    ### 1.8 - Neural Networks go brrrr.....
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
                fp_batch = apply_network(X_batch, trained_net)

                # backward pass
                grad_batch = backward_pass(
                    X_batch, Y_batch, fp_batch, trained_net, lam
                )

                # update parameters
                for i in range(len(trained_net["W"])):
                    trained_net["W"][i] -= eta * grad_batch["W"][i]
                    trained_net["b"][i] -= eta * grad_batch["b"][i]

            # compute metrics after each epoch
            # training metrics
            fp_train = apply_network(X, trained_net)
            P_train = fp_train["P"]
            # loss
            train_loss = compute_loss(P_train, y)
            train_loss_output.append(train_loss)
            # cost
            train_cost = compute_cost(P_train, y, trained_net, lam)
            train_cost_output.append(train_cost)
            # accuracy
            train_acc = compute_accuracy(P_train, y)
            train_acc_output.append(train_acc)

            # validation metrics
            fp_valid = apply_network(X_val, trained_net)
            P_valid = fp_valid["P"]
            # loss
            valid_loss = compute_loss(P_valid, y_val)
            valid_loss_output.append(valid_loss)
            # cost
            valid_cost = compute_cost(P_valid, y_val, trained_net, lam)
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
        print()
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
        fig, axs = plt.subplots(2, 10, figsize=(15, 6))

        # Layer 1 -> input to hidden layer
        print("Layer 1 weights (input to hidden):")
        for i in range(10):
            w_im = trained_net["W"][0][i, :].reshape(32, 32, 3, order="F")
            w_im = np.transpose(w_im, (1, 0, 2))
            w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
            axs[0, i].imshow(w_im_norm)
            axs[0, i].set_title(f"Hidden Neuron {i}")
            axs[0, i].axis("off")

        # Layer 2 -> hidden layer to output
        print("Layer 2 weights (hidden to output):")
        for i in range(10):
            w_i = trained_net["W"][1][i, :].reshape(5, 10)
            w_norm = (w_i - np.min(w_i)) / (np.max(w_i) - np.min(w_i))
            axs[1, i].imshow(w_norm, cmap="viridis")
            axs[1, i].set_title(f"Output Class {i}")
            axs[1, i].axis("off")

        plt.tight_layout()

        if filename:
            plt.savefig(
                filename + "_intermediate_results.png",
                dpi=300,
                bbox_inches="tight",
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
        fp_test = apply_network(norm_X_test, trained_net)
        P_test = P_test = fp_test["P"]
        test_loss = compute_loss(P_test, y_test_actual_labels)
        test_accuracy = compute_accuracy(P_test, y_test_actual_labels)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
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
            plt.savefig(filename + "_results.png", dpi=300, bbox_inches="tight")
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


    def run_experiments(GDparams, rng):
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
        (
            _,
            _,
        ) = test_trained_network(trained_net)
        print()

        print(f"Display the representations learned by the neural network...")
        display_weight_matrix(trained_net)
        print()

        print(f"Plot loss and accuracy...")
        plot_results(
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
        )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercise 2 - Sanity Check
    """)
    return


@app.cell
def _(
    Y_train_labels,
    init_net,
    initialize_network,
    mini_batch_gd,
    norm_X_train,
    np,
    y_train_actual_labels,
):
    def sanity_check_experiment(
        norm_X_train, Y_train_labels, y_train_actual_labels, num_examples=100
    ):
        # create a subset of the data based on the number of specified examples
        X_small = norm_X_train[:, :num_examples]
        Y_small = Y_train_labels[:, :num_examples]
        y_small = y_train_actual_labels[:num_examples]

        # initialize parameters
        input_dim = X_small.shape[0]
        hidden_dim = 50
        output_dim = 10
        rng = np.random.default_rng(42)
        small_net = initialize_network(input_dim, hidden_dim, output_dim, rng)

        # define training parameters with no regularization(lam = 0)
        GDparams = {"n_batch": 100, "eta": 0.01, "n_epochs": 200, "lam": 0}

        # train the network
        print(f"Training a small network for testing...")
        print(f"batch size: {GDparams['n_batch']}")
        print(f"learning rate: {GDparams['eta']}")
        print(f"number of epochs: {GDparams['n_epochs']}")
        print(f"regularization: {GDparams['lam']}")
        print()
        trained_net, train_loss_output, _, _, _, train_acc_output, _ = (
            mini_batch_gd(
                X_small,
                Y_small,
                y_small,
                X_small,
                y_small,
                GDparams,
                init_net,
                rng,
            )
        )

        # check final results from training loss and training accuracy
        final_train_loss = train_loss_output[-1]
        final_train_acc = train_acc_output[-1]
        print(
            f"Final training loss: {final_train_loss:.4f}, Final training accuracy: {final_train_acc:.4f}"
        )


    sanity_check_experiment(
        norm_X_train, Y_train_labels, y_train_actual_labels, 100
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercise 3 - Cyclical Learning Rates
    """)
    return


@app.cell
def _(np):
    def compute_cyclical_learning_rate(t, eta_min, eta_max, n_s):
        # l - cycle counter. 1 cycle = 2 * ns
        l = np.floor(t / (2 * n_s))
        # calculate the learning rate for the first half of the cycle
        if 2 * l * n_s <= t <= (2 * l + 1) * n_s:
            return eta_min + ((t - 2 * l * n_s) / n_s) * (eta_max - eta_min)
            # calculate the learning rate for the second half of the cycle
        elif (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s:
            return eta_max - ((t - (2 * l + 1) * n_s) / n_s) * (eta_max - eta_min)
    return (compute_cyclical_learning_rate,)


@app.cell
def _(
    apply_network,
    backward_pass,
    compute_accuracy,
    compute_cost,
    compute_cyclical_learning_rate,
    compute_loss,
    copy,
):
    """
    Train the network using mini-batch gradient descent.
    """


    def mini_batch_gd_cyclical(X, Y, y, X_val, y_val, GDparams, init_net, rng):
        # Deep copy to avoid modifying the original network
        trained_net = copy.deepcopy(init_net)
        n_batch = GDparams["n_batch"]
        n_epochs = GDparams["n_epochs"]
        lam = GDparams["lam"]

        # cyclical learning rate
        # min learning rate value
        eta_min = GDparams["eta_min"]
        # max learning rate value
        eta_max = GDparams["eta_max"]
        # step size - controls the number of steps to complete 1 cycle
        n_s = GDparams["n_s"]

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

        # record t steps
        t_steps = []
        t_step = 0

        # training loop
        for epoch in range(n_epochs):
            # shuffle training data
            shuffle_indices = rng.permutation(n)
            X_shuffled = X[:, shuffle_indices]
            Y_shuffled = Y[:, shuffle_indices]
            y_shuffled = y[shuffle_indices]

            # batch processing
            # process n / n_batch images at a time
            for j in range(n // n_batch):
                j_start = j * n_batch
                j_end = (j + 1) * n_batch
                X_batch = X_shuffled[:, j_start:j_end]
                Y_batch = Y_shuffled[:, j_start:j_end]

                # compute learning rate
                t_step += 1
                eta = compute_cyclical_learning_rate(t_step, eta_min, eta_max, n_s)

                # forward pass
                fp_batch = apply_network(X_batch, trained_net)

                # backward pass
                grad_batch = backward_pass(
                    X_batch, Y_batch, fp_batch, trained_net, lam
                )

                # update parameters
                for i in range(len(trained_net["W"])):
                    trained_net["W"][i] -= eta * grad_batch["W"][i]
                    trained_net["b"][i] -= eta * grad_batch["b"][i]

            # compute metrics after each epoch
            # training metrics
            fp_train = apply_network(X, trained_net)
            P_train = fp_train["P"]
            # loss
            train_loss = compute_loss(P_train, y)
            train_loss_output.append(train_loss)
            # cost
            train_cost = compute_cost(P_train, y, trained_net, lam)
            train_cost_output.append(train_cost)
            # accuracy
            train_acc = compute_accuracy(P_train, y)
            train_acc_output.append(train_acc)

            # validation metrics
            fp_valid = apply_network(X_val, trained_net)
            P_valid = fp_valid["P"]
            # loss
            valid_loss = compute_loss(P_valid, y_val)
            valid_loss_output.append(valid_loss)
            # cost
            valid_cost = compute_cost(P_valid, y_val, trained_net, lam)
            valid_cost_output.append(valid_cost)
            # accuracy
            valid_acc = compute_accuracy(P_valid, y_val)
            valid_acc_output.append(valid_acc)

            # t steps
            t_steps.append(t_step)

            print(
                f"Epoch {epoch + 1}/{n_epochs}, "
                f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, "
                f"Training Cost: {train_cost:.4f}, Validation Cost: {valid_cost:.4f}, "
                f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}\n"
            )
        return (
            trained_net,
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
            t_steps,
        )
    return (mini_batch_gd_cyclical,)


@app.cell
def _(plt):
    def plot_cyclical_results(
        filename,
        train_loss_output,
        valid_loss_output,
        train_cost_output,
        valid_cost_output,
        train_acc_output,
        valid_acc_output,
        t_steps,
    ):
        # Create a figure with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        # loss plot
        axes[0, 0].plot(t_steps, train_loss_output, label="Training Loss")
        axes[0, 0].plot(t_steps, valid_loss_output, label="Validation Loss")
        axes[0, 0].set_xlabel("Update Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].set_title("Training and Validation Loss - Update Steps")
        axes[0, 0].grid(True)

        # cost plot
        axes[0, 1].plot(t_steps, train_cost_output, label="Training Cost")
        axes[0, 1].plot(t_steps, valid_cost_output, label="Validation Cost")
        axes[0, 1].set_xlabel("Update Step")
        axes[0, 1].set_ylabel("Cost")
        axes[0, 1].legend()
        axes[0, 1].set_title("Training and Validation Cost - Update Steps")
        axes[0, 1].grid(True)

        # accuracy plot
        axes[0, 2].plot(t_steps, train_acc_output, label="Training Accuracy")
        axes[0, 2].plot(t_steps, valid_acc_output, label="Validation Accuracy")
        axes[0, 2].set_xlabel("Update Step")
        axes[0, 2].set_ylabel("Accuracy")
        axes[0, 2].legend()
        axes[0, 2].set_title("Training and Validation Accuracy - Update Steps")
        axes[0, 2].grid(True)

        # loss plot
        axes[1, 0].plot(train_loss_output, label="Training Loss")
        axes[1, 0].plot(valid_loss_output, label="Validation Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].set_title("Training and Validation Loss")
        axes[1, 0].grid(True)

        # cost plot
        axes[1, 1].plot(train_cost_output, label="Training Cost")
        axes[1, 1].plot(valid_cost_output, label="Validation Cost")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Cost")
        axes[1, 1].legend()
        axes[1, 1].set_title("Training and Validation Cost")
        axes[1, 1].grid(True)

        # accuracy plot
        axes[1, 2].plot(train_acc_output, label="Training Accuracy")
        axes[1, 2].plot(valid_acc_output, label="Validation Accuracy")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Accuracy")
        axes[1, 2].legend()
        axes[1, 2].set_title("Training and Validation Accuracy")
        axes[1, 2].grid(True)

        plt.tight_layout()
        if filename:
            plt.savefig(filename + "_results.png", dpi=300, bbox_inches="tight")
        plt.show()
    return (plot_cyclical_results,)


@app.cell
def _(
    Y_train_labels,
    display_weight_matrix,
    init_net,
    mini_batch_gd_cyclical,
    norm_X_train,
    norm_X_valid,
    plot_cyclical_results,
    test_trained_network,
    y_train_actual_labels,
    y_valid_actual_labels,
):
    def run_cyclical_experiments(filename, GDparams, rng):
        print(f"Run cyclical experiment...")
        print(f"batch size: {GDparams['n_batch']}")
        print(f"eta_min: {GDparams['eta_min']}")
        print(f"eta_max: {GDparams['eta_max']}")
        print(f"n_s: {GDparams['n_s']}")
        print(f"number of epochs: {GDparams['n_epochs']}")
        print(f"lam: {GDparams['lam']}")
        print()
        print("Train neural network with cyclical learning rates...")
        (
            trained_net,
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
            t_steps,
        ) = mini_batch_gd_cyclical(
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
        plot_cyclical_results(
            filename,
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
            t_steps,
        )
    return (run_cyclical_experiments,)


@app.cell
def _(np, run_cyclical_experiments):
    def cyclical_experiments():
        GDparams = {
            "n_batch": 100,
            "eta_min": 1e-5,
            "eta_max": 1e-1,
            "n_s": 500,
            "n_epochs": 10,
            "lam": 0.01,
        }
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_cyclical_experiments("initial_experiments", GDparams, rng)


    cyclical_experiments()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exercise 4 - Training go brrrr.....
    """)
    return


@app.cell
def _(np, run_cyclical_experiments):
    def experiment1():
        GDparams = {
            "n_batch": 100,
            "eta_min": 1e-5,
            "eta_max": 1e-1,
            "n_s": 800,
            "n_epochs": 48,
            "lam": 0.01,
        }
        # set up random number generator for reproducibility
        rng = np.random.default_rng(42)
        run_cyclical_experiments("Train 3 Cycles", GDparams, rng)


    experiment1()
    return


@app.cell
def _(load_batch, np):
    def load_all_data(validation_set_size=5000):
        data_files = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ]

        # construct the entire dataset
        X, Y, y = load_batch(data_files[0])
        for i in range(1, 5):
            X_batch, Y_batch, y_batch = load_batch(data_files[i])
            X = np.concatenate((X, X_batch), axis=1)
            Y = np.concatenate((Y, Y_batch), axis=1)
            y = np.concatenate((y, y_batch))

        # randomly split data into train/validation datasets
        n = X.shape[1]
        indices = np.random.permutation(n)
        training_set_idx = indices[validation_set_size:]
        validation_set_idx = indices[:validation_set_size]

        X_train = X[:, training_set_idx]
        Y_train = Y[:, training_set_idx]
        y_train = y[training_set_idx]

        X_valid = X[:, validation_set_idx]
        Y_valid = Y[:, validation_set_idx]
        y_valid = y[validation_set_idx]

        return X_train, Y_train, y_train, X_valid, Y_valid, y_valid
    return (load_all_data,)


@app.cell
def _(load_all_data, load_batch):
    print(f"load all data and split into training and testing data...")
    (
        raw_X_train_all,
        Y_train_all,
        y_train_all,
        raw_X_valid_all,
        Y_valid_all,
        y_valid_all,
    ) = load_all_data(5000)
    print(f"load all test data...")
    raw_X_test, Y_test, y_test = load_batch("test_batch")
    return (
        Y_test,
        Y_train_all,
        Y_valid_all,
        raw_X_test,
        raw_X_train_all,
        raw_X_valid_all,
        y_test,
        y_train_all,
        y_valid_all,
    )


@app.cell
def _(Y_train_all, raw_X_train_all, y_train_all):
    print(f"X_train_all dimensions -> {raw_X_train_all.shape}")
    print(f"Y_train_all dimensions -> {Y_train_all.shape}")
    print(f"y_train_all dimensions -> {y_train_all.shape}")
    return


@app.cell
def _(Y_valid_all, raw_X_valid_all, y_valid_all):
    print(f"X_valid_all dimensions -> {raw_X_valid_all.shape}")
    print(f"Y_valid_all dimensions -> {Y_valid_all.shape}")
    print(f"y_valid_all dimensions -> {y_valid_all.shape}")
    return


@app.cell
def _(Y_test, raw_X_test, y_test):
    print(f"X test dimensions -> {raw_X_test.shape}")
    print(f"Y test dimensions -> {Y_test.shape}")
    print(f"y test dimensions -> {y_test.shape}")
    return


@app.cell
def _():
    print(f"preprocess the training, validation and test datasets...")
    # norm_X_train_all, norm_X_valid_all, norm_X_test_all = preprocess_data(raw_X_train_all, raw_X_valid_all, raw_X_test)
    return


@app.cell
def _(
    apply_network,
    compute_accuracy,
    initialize_network,
    mini_batch_gd_cyclical,
    np,
    test_trained_network,
):
    def coarse_lambda_search(
        X_train, Y_train, y_train, X_valid, Y_valid, y_valid, rng, hidden_dim=50
    ):
        l_min, l_max = -5, -1
        # number of lambda values to test
        num_lambdas = 8
        # define a uniform range to sample lambda values
        lambda_values = np.logspace(l_min, l_max, num_lambdas)

        # set up parameters
        input_dim = X_train.shape[0]
        output_dim = 10
        n_batch = 100
        n = X_train.shape[1]
        n_s = 2 * np.floor(n / n_batch)

        # store results
        results = []
        # count number of lambda values
        iter_count = 0

        for lam in lambda_values:
            print(f"lambda search {iter_count + 1}...")
            print(f"lam value: {lam:.6f}")
            init_net = initialize_network(input_dim, hidden_dim, output_dim, rng)
            GDparams = {
                "n_batch": n_batch,
                "eta_min": 1e-5,
                "eta_max": 1e-1,
                "n_s": n_s,
                "n_epochs": int(2 * 2 * n_s / (n / n_batch)),
                "lam": lam,
            }
            trained_net, _, _, _, _, _, _, _ = mini_batch_gd_cyclical(
                X_train,
                Y_train,
                y_train,
                X_valid,
                y_valid,
                GDparams,
                init_net,
                rng,
            )
            fp_valid = apply_network(X_valid, trained_net)
            P_valid = fp_valid["P"]
            valid_accuracy = compute_accuracy(P_valid, y_valid)
            test_loss, test_accuracy = test_trained_network(trained_net)
            print(
                f"lambda: {lam:.6f}, validation accuracy: {valid_accuracy:.4f}, test accuracy: {test_accuracy:.4f}"
            )
            results.append((lam, valid_accuracy, test_accuracy))
            iter_count += 1
            print()

        return results
    return (coarse_lambda_search,)


@app.function
def find_top_n_lambdas(lambda_results, n=3, output_file=None):
    for lam, valid_accuracy, test_accuracy in lambda_results:
        print(
            f"lam: {lam}, validation accuracy: {valid_accuracy}, test_accuracy: {test_accuracy}"
        )
    print()

    print(f"sort results from greatest to least...")
    sorted_lambda_results = sorted(
        lambda_results, key=lambda x: x[2], reverse=True
    )
    for lam, valid_accuracy, test_accuracy in sorted_lambda_results:
        print(
            f"lam: {lam}, validation accuracy: {valid_accuracy}, test_accuracy: {test_accuracy}"
        )
    print()

    print(f"Top {n} lambda results...")
    top_n_lambda_results = sorted_lambda_results[:n]
    for lam, valid_accuracy, test_accuracy in top_n_lambda_results:
        print(
            f"lam: {lam}, validation accuracy: {valid_accuracy}, test accuracy: {test_accuracy}"
        )
    print()

    if output_file:
        with open(output_file, "w") as f:
            f.write(
                f"All lambda results - number of lambdas: {len(lambda_results)}...\n"
            )
            for lam, valid_accuracy, test_accuracy in lambda_results:
                f.write(
                    f"lam: {lam}, validation accuracy: {valid_accuracy}, test_accuracy: {test_accuracy}\n"
                )

            f.write("\nSorted lambda results (greatest to least)...\n")
            for lam, valid_accuracy, test_accuracy in sorted_lambda_results:
                f.write(
                    f"lam: {lam}, validation accuracy: {valid_accuracy}, test_accuracy: {test_accuracy}\n"
                )

            f.write(f"\nTop {n} lambda results...\n")
            for lam, valid_accuracy, test_accuracy in top_n_lambda_results:
                f.write(
                    f"lam: {lam}, validation accuracy: {valid_accuracy}, test accuracy: {test_accuracy}\n"
                )

    return top_n_lambda_results


@app.cell
def _(
    Y_train_all,
    Y_valid_all,
    coarse_lambda_search,
    norm_X_train_all,
    norm_X_valid_all,
    np,
    y_train_all,
    y_valid_all,
):
    print(f"Coarse Lambda Search...")
    rng = np.random.default_rng(42)
    coarse_lambda_results = coarse_lambda_search(
        norm_X_train_all,
        Y_train_all,
        y_train_all,
        norm_X_valid_all,
        Y_valid_all,
        y_valid_all,
        rng,
    )
    return (coarse_lambda_results,)


@app.cell
def _(coarse_lambda_results):
    print(f"Display coarse lambda results...")
    print(f"length of coarse lambda results: {len(coarse_lambda_results)}")
    top_coarse_lambda_results = find_top_n_lambdas(
        coarse_lambda_results, 3, output_file="coarse-lambda-results.txt"
    )
    return (top_coarse_lambda_results,)


@app.cell
def _(
    apply_network,
    compute_accuracy,
    initialize_network,
    mini_batch_gd_cyclical,
    np,
    test_trained_network,
):
    def fine_lambda_search(
        X_train,
        Y_train,
        y_train,
        X_valid,
        Y_valid,
        y_valid,
        rng,
        top_lambda_results,
        hidden_dim=50,
    ):
        # define min and max lambda
        min_lambda = min(top_lambda_results, key=lambda x: x[1])
        max_lambda = max(top_lambda_results, key=lambda x: x[1])
        print(f"min_lambda value: {min_lambda[0]}")
        print(f"max_lambda value: {max_lambda[0]}")

        # define search range
        margin = 0.5
        l_min = np.log10(min_lambda[0]) + margin
        l_max = np.log10(max_lambda[0]) + margin
        print(f"Search range: 10^{l_min:.6f} to 10^{l_max:.6f}")
        # number of lambdas to test for
        num_lambdas = 8
        # define a uniform range to search for lambdas
        lambda_values = np.logspace(l_min, l_max, num_lambdas)

        # set up parameters
        input_dim = X_train.shape[0]
        output_dim = 10
        n_batch = 100
        n = X_train.shape[1]
        n_s = 2 * np.floor(n / n_batch)

        # store results
        results = []
        # count number of lambda values
        iter_count = 0

        for lam in lambda_values:
            print(f"lambda search {iter_count + 1}")
            print(f"lam value: {lam:.6f}")
            init_net = initialize_network(input_dim, hidden_dim, output_dim, rng)
            GDparams = {
                "n_batch": n_batch,
                "eta_min": 1e-5,
                "eta_max": 1e-1,
                "n_s": n_s,
                "n_epochs": int(4 * 2 * n_s / (n / n_batch)),
                "lam": lam,
            }
            trained_net, _, _, _, _, _, _, _ = mini_batch_gd_cyclical(
                X_train,
                Y_train,
                y_train,
                X_valid,
                y_valid,
                GDparams,
                init_net,
                rng,
            )
            fp_valid = apply_network(X_valid, trained_net)
            P_valid = fp_valid["P"]
            valid_accuracy = compute_accuracy(P_valid, y_valid)
            test_loss, test_accuracy = test_trained_network(trained_net)
            print(
                f"lambda: {lam:.6f}, validation accuracy: {valid_accuracy:.6f}, test accuracy: {test_accuracy:.6f}"
            )
            results.append((lam, valid_accuracy, test_accuracy))
            iter_count += 1
            print()

        return results
    return (fine_lambda_search,)


@app.cell
def _(
    Y_train_all,
    Y_valid_all,
    fine_lambda_search,
    norm_X_train_all,
    norm_X_valid_all,
    np,
    top_coarse_lambda_results,
    y_train_all,
    y_valid_all,
):
    print(f"Fine Lambda Search...")
    fine_lambda_results = fine_lambda_search(
        norm_X_train_all,
        Y_train_all,
        y_train_all,
        norm_X_valid_all,
        Y_valid_all,
        y_valid_all,
        np.random.default_rng(42),
        top_coarse_lambda_results,
    )
    return (fine_lambda_results,)


@app.cell
def _(fine_lambda_results):
    print(f"Display fine lambda results...")
    print(f"length of fine lambda results: {len(fine_lambda_results)}")
    top_fine_lambda_results = find_top_n_lambdas(
        fine_lambda_results, 3, output_file="fine-lambda-search.txt"
    )
    return (top_fine_lambda_results,)


@app.cell
def _(
    Y_train_all,
    Y_valid_all,
    fine_lambda_search,
    norm_X_train_all,
    norm_X_valid_all,
    np,
    top_fine_lambda_results,
    y_train_all,
    y_valid_all,
):
    print(f"Fine Lambda Search 2...")
    fine_lambda_results_2 = fine_lambda_search(
        norm_X_train_all,
        Y_train_all,
        y_train_all,
        norm_X_valid_all,
        Y_valid_all,
        y_valid_all,
        np.random.default_rng(42),
        top_fine_lambda_results,
    )
    return (fine_lambda_results_2,)


@app.cell
def _(fine_lambda_results_2):
    print(f"Display fine lambda results 2...")
    print(f"length of fine lambda results 2: {len(fine_lambda_results_2)}")
    top_fine_lambda_results_2 = find_top_n_lambdas(
        fine_lambda_results_2, 3, output_file="fine-lambda-search2.txt"
    )
    return (top_fine_lambda_results_2,)


@app.cell
def _(mo):
    mo.md("""
    Exercise 5 - Optimized Network goes brrrr.......
    """)
    return


@app.cell
def _():
    print(f"load all data and split into training and testing data...")
    # raw_X_train_all, Y_train_all, y_train_all, raw_X_valid_all, Y_valid_all, y_valid_all = load_all_data(1000)
    print(f"load all test data...")
    # raw_X_test, Y_test, y_test = load_batch("test_batch")
    return


@app.cell
def _(Y_train_all, raw_X_train_all, y_train_all):
    print(f"X_train_all dimensions: {raw_X_train_all.shape}")
    print(f"Y_train_all dimensions: {Y_train_all.shape}")
    print(f"y_train_all dimensions: {y_train_all.shape}")
    return


@app.cell
def _(Y_valid_all, raw_X_valid_all, y_valid_all):
    print(f"X_valid_all dimensions: {raw_X_valid_all.shape}")
    print(f"Y_valid_all dimensions: {Y_valid_all.shape}")
    print(f"y_valid_all dimensions: {y_valid_all.shape}")
    return


@app.cell
def _(Y_test, raw_X_test, y_test):
    print(f"X test dimensions: {raw_X_test.shape}")
    print(f"Y test dimensions: {Y_test.shape}")
    print(f"y test dimensions: {y_test.shape}")
    return


@app.cell
def _(preprocess_data, raw_X_test, raw_X_train_all, raw_X_valid_all):
    print(f"preprocess the training, validation and test datasets...")
    norm_X_train_all, norm_X_valid_all, norm_X_test_all = preprocess_data(
        raw_X_train_all, raw_X_valid_all, raw_X_test
    )
    return norm_X_train_all, norm_X_valid_all


@app.cell
def _(top_fine_lambda_results_2):
    optimal_lam = top_fine_lambda_results_2[0][0]
    print(f"lam value: {optimal_lam:.6f}")
    return (optimal_lam,)


@app.cell
def _(
    display_weight_matrix,
    initialize_network,
    mini_batch_gd_cyclical,
    np,
    plot_cyclical_results,
    test_trained_network,
):
    def optimize_lambda_train(
        optimal_lam,
        X_train,
        Y_train,
        y_train,
        X_valid,
        Y_valid,
        y_valid,
        rng,
        hidden_dim=50,
    ):
        # set up parameters
        input_dim = X_train.shape[0]
        output_dim = 10
        n_batch = 100
        lam = optimal_lam
        n = X_train.shape[1]
        n_s = 2 * np.floor(n / n_batch)

        # init network
        init_net = initialize_network(input_dim, hidden_dim, output_dim, rng)

        # set up params
        GDparams = {
            "n_batch": n_batch,
            "eta_min": 1e-5,
            "eta_max": 1e-1,
            "n_s": n_s,
            "n_epochs": int(3 * 2 * n_s / (n / n_batch)),
            "lam": lam,
        }

        # train the network
        print("Train neural network with cyclical learning rates...")
        (
            trained_net,
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
            t_steps,
        ) = mini_batch_gd_cyclical(
            X_train, Y_train, y_train, X_valid, y_valid, GDparams, init_net, rng
        )
        print()

        # test against against ground truth
        print(f"Test trained neural network against ground truth...")
        test_trained_network(trained_net)
        print()

        print(f"Display the representations learned by the neural network...")
        display_weight_matrix("optimized_lambda", trained_net)
        print()

        # plot results
        print(f"Plot loss and accuracy...")
        plot_cyclical_results(
            "optimized_lambda",
            train_loss_output,
            valid_loss_output,
            train_cost_output,
            valid_cost_output,
            train_acc_output,
            valid_acc_output,
            t_steps,
        )
    return (optimize_lambda_train,)


@app.cell
def _(
    Y_train_all,
    Y_valid_all,
    norm_X_train_all,
    norm_X_valid_all,
    np,
    optimal_lam,
    optimize_lambda_train,
    y_train_all,
    y_valid_all,
):
    optimize_lambda_train(
        optimal_lam,
        norm_X_train_all,
        Y_train_all,
        y_train_all,
        norm_X_valid_all,
        Y_valid_all,
        y_valid_all,
        rng=np.random.default_rng(42),
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
