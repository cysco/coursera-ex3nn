Hi everyone!

Here is a Python transcript of ex3_nn.m (and all its related subroutines) from Andrew Ng Machine Learning class on Coursera. This is the classical example of digit recognition with neural networks.

Link to the course: https://www.coursera.org/learn/machine-learning/

The program was originally written in Octave/Matlab, with some blanks to fill. I sticked to their structure, variable names and so on. I am conscious there are more straightforward way to solve this problem, using for instance keras to build your neural network. But keep in mind that the goal of this exercise is to 'hard code' your neural network and get a grip of what happens inside (initial random weights, gradient descent, backpropagation and so on).

I included the two data files ex3data1.mat and ex3weights.mat that you can retrieve from Coursera (programming assignment - week 4). Access to the lessons and the data is free of charge; payment is only required if you want the certificate.

I implemented various comments to help following what happens in the code, and to explain some adjustments I made with respect to the original one. If something is unclear, please let me know and I'll do my best to explain it better.

Notice that I do not detail the principles of the neural network algorithm; I assume that you attended the lectures on Coursera and/or
have some knowledge about what happens numerically from one layer to another (essentially matrix products).

Other exercises from the same course will come. Thanks for your interest and stay tuned!

~ Cysco ~