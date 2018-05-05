import random
import numpy as np

class Network(object):

    def __init__(self, no_of_neurons):
        

        self.num_layers = len(no_of_neurons)
        self.no_of_neurons = no_of_neurons
        # bias and weights generated randomly
        self.biases = [np.random.randn(y, 1) for y in no_of_neurons[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(no_of_neurons[:-1], no_of_neurons[1:])]

    def feedforward(self, a):
        
        # sigmoid actication function
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def descent(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # Train the neural network using mini-batch stochastic
        # gradient descent.  
        # network will be evaluated against the test data after each
        # epoch, and partial progress printed out. 

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_del_b, delta_del_w = self.backprop(x, y)
            del_b = [nb+dnb for nb, dnb in zip(del_b, delta_del_b)]
            del_w = [nw+dnw for nw, dnw in zip(del_w, delta_del_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, del_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, del_b)]

    def backprop(self, x, y):
        # Return a tuple ``(del_b, del_w)`` representing the
        # gradient for the cost function C_x.  ``del_b`` and
        # ``del_w`` are layer-by-layer lists of numpy arrays, similar
        # to ``self.biases`` and ``self.weights``.
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            del_b[-l] = delta
            del_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (del_b, del_w)

    def evaluate(self, test_data):
        # returns no of cases where prediction is accurate
        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y)/10000s for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        
        # Return the vector of partial derivatives 
        # for the output activations.
        return (output_activations-y)


def sigmoid(z):
    # Sigmoid function
    return 1.0/(1.0+np.exp(-z))

# derivative of sigmoid function
def sigmoid_prime(p): 
    return sigmoid(p)*(1-sigmoid(p))
