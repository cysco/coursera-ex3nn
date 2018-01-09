## Initialization : importing the relevant modules and packages
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

## Setting up the parameters that will be used for the exercise
input_layer_size  = 400		# Each image is 20x20, so 400 input units are required, one for each digit
hidden_layer_size = 25		# 25 hidden units
num_labels = 10				# 10 labels (from 0 to 9) for 10 output units


## ===================================================================
## ================= FUNCTION DISPLAYDATA(X, EXAMPLE_WIDTH) ==========
##
## The function displays 2D data (images) stored in X in a nice grid.
## All the images are black and white.
## The size of the grid depends on the example_width parameter.
##
## ===================================================================
def displayData(X, example_width=None):
    # If the input X is an n-elements 1D array (X.shape = (n,)),
    # reshape it as 2D array (X.shape = (1, n)).
    # This is necessary to perform the next steps.
    if len(X.shape) == 1:
        X.shape = (1, len(X))

    # Setting example_width automatically if not passed in
    if example_width is None:
        example_width = int(np.round(np.sqrt(X.shape[1])))

    # m = number of rows in X (i.e. number of examples)
    # n = number of columns in X (i.e. number of digits per example)
    m, n = X.shape
    example_height = np.int(np.round(n / example_width))

    # Computing the grid size from the number of examples to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setting up a blank display
    # The display size depends on i) the grid size and ii) the size of each individual example.
    # The padding in between each example is also taken into account for the computation.
    display_array = np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

    # Copying each example into a patch on the display array
    # A loop is performed in order to 'attach' each example (numbered curr_ex) to each patch (numbered [i, j]).
    curr_ex = 0
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            if curr_ex > m:
                break

            # Getting the max value of the patch (for normalization purpose)
            max_val = max(np.abs(X[curr_ex, :]))

            # Setting up the borders of the patch
            down = pad + j * (example_height + pad)
            up = pad + j * (example_height + pad) + example_height
            left = pad + i * (example_width + pad)
            right =  pad + i * (example_width + pad) + example_width

            # Reshaping the example in order to match the size of the patch
            # In the exercise, curr_X.shape is initially 1x400, reshaped into
            # an 20x20 array and then normalized.
            curr_X = X[curr_ex, :]
            curr_X.shape = (example_height, example_width)
            curr_X = curr_X / max_val

            # Assigning the values of curr_X to the current patch
            # Notice that we take 'curr_X transposed' in order to display
            # the image(s) in the proper direction.
            display_array[down:up, left:right] = curr_X.T

            curr_ex = curr_ex + 1

        if curr_ex > m:
            break

    plt.imshow(display_array, aspect='auto', cmap='Greys')
    plt.axis('off')
    plt.show()


## ===========================================================================
## ================= FUNCTION PREDICT(THETA1, THETA2, X) =====================
##
## The function predicts the label of an input given a trained neural network.
## The output is the predicted label of X given the trained weights of a
## neural network (Theta1, Theta2).
##
## ===========================================================================
def predict(Theta1, Theta2, X):
    # If the input X is an n-elements 1D array (X.shape = (n,)),
    # reshape it as 2D array (X.shape = (1, n)).
    # This is necessary to perform the next steps.
    if len(X.shape) == 1:
        X.shape = (1, len(X))

    # m is the number of rows in X (i.e. the number of examples) 
    m = X.shape[0]

    # Initializing the probability matrix p that will be returned at the end
    # p is a column vector containing the predicted value from 0 to 9
    p = np.zeros((m, 1))

    # Adding a column of '1s' to X
    # Each layer of the neural network has a 'bias' unit, which consists in
    # an m-elements column vector.
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    z2 = X @ Theta1.T
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    # IMPORTANT: the images stored in X represent numbers from 0 to 9.
    # Octave/Matlab arrays are indexed starting from 1, so they adopted
    # the convention that y = 10 whenever X represents a zero.
    # Since Python arrays are indexed starting from 0, it was necessary
    # to increment p by 1 in order to obtain proper predictions.
    p = np.argmax(a3, axis=1)
    p = 1 + p
    return p


## =============================================================
## ================= FUNCTION SIGMOID(Z) =======================
##
## Simply computing the sigmoid function for an input matrix z
##
## =============================================================
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g


## ===============================================================
## ================= MAIN PROGRAM ================================
##
## =========== Part 1: Loading and Visualizing Data ==============
## The exercise starts by loading and visualizing the dataset,
## which consists in handwritten digits from 0 to 9.
##
## ===============================================================
print('Loading and visualizing the data...\n')

# Loading the original Matlab data into the Python arrays X and y
matdata = sio.loadmat('ex3data1.mat')
X = matdata['X']
y = matdata['y']
y.shape = (5000, )

# m = 5000 is the number of entries in the dataset
m = X.shape[0]

# Randomly selecting 100 data points to display
sel = [i for i in range(m)]
shuffle(sel)
sel = sel[0:100]

displayData(X[sel, :])
input('Program paused. Press enter to continue.\n')

## ============================================================
## ================ Part 2: Loading Pameters ==================
##
## In this part of the exercise, we load some pre-initialized
## neural network parameters.
##
## ============================================================
print('Loading Saved Neural Network Parameters ...\n')

# Loading the original Matlab weights into the Python arrays
# Theta1 and Theta2
matweights = sio.loadmat('ex3weights')
Theta1 = matweights['Theta1']
Theta2 = matweights['Theta2']

## =======================================================================
## ================= PART 3: IMPLEMENT PREDICT ===========================
##
##  After training the neural network, we would like to use it to predict
##  the labels. You will now implement the "predict" function to use the
##  neural network to predict the labels of the training set. This lets
##  you compute the training set accuracy.
##
## =======================================================================
pred = predict(Theta1, Theta2, X)
true_pos = pred == y

print('Training Set Accuracy: %2.2f\n' % (np.mean(true_pos == True) * 100.))
input('Program paused. Press enter to continue.\n')

# To give you an idea of the network's output, you can also run
# through the examples one at a time to see what it is predicting.

# Randomly permuting examples
rp = [i for i in range(m)]
shuffle(rp)

for i in rp:
    print('\nDisplaying Example Image\n')
    displayData(X[i, :])

    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: %d (digit %d)\n' % (pred, y[i]))
    
    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q': break

## ================= THIS IS THE END =====================================
## =======================================================================