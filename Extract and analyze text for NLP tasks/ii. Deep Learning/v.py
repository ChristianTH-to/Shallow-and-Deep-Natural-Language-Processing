# Softmax, LeakyReLU and Sigmoid function plots


# Dependencies

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
%matplotlib inline

# Softmax

x = np.arange(-5, 5, 0.01)
y = np.exp(x) / float(sum(np.exp(x)))
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x,y)
plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()

# LeakyreLU

import numpy as np

class Activation(object):
    def __init__(self, x):
        self.x = x
        self.p = None
        self.derivative = None

    def forward(self):
        pass

    def backward(self):
        pass

    def __call__(self):
        res = list()
        res.append(self.forward())
        res.append(self.backward())
        return res

class LeakyRelu(Activation):
    def __init__(self, x, alpha=0.1):
        super(LeakyRelu, self).__init__(x)
        self.alpha = alpha

    def forward(self):
        self.p = np.maximum(self.alpha * self.x, self.x)
        return self.p

    #def backward(self):
     #   self.derivative = np.full_like(self.p, 1)
      #  self.derivative[self.p < 0] = self.alpha
       # return self.derivative
if __name__ == "__main__":
    figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
    x = np.linspace(-10, 10, 500)
    plt.plot(x, LeakyRelu(x)()[0], label='leakyRelu_forward')
    plt.legend(loc='best')
    plt.title('ReLU Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()

# Sigmoid

x = np.arange(-5,5,0.01)
y = 1 / (1+np.exp(-x))
#ds = y * (1 - y) # Derivative of sigmoid
#ds = (y(x+step) - y(x)) / step
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x,y)
plt.title('Sigmoid Function Derivative')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
