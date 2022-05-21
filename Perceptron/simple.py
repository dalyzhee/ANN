import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
# loading data 
text = np.loadtxt('simple.txt')

data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))


plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# define maximum values for each dimension
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

# number of neuron
num_output = labels.shape[1]

# define perceptron with 2 input neuron
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

perceptron = nl.net.newp([dim1, dim2], num_output)

# train data
error_prog = perceptron.train(data, labels, epochs=100, show=20, lr=0.03)

# plot training progress
plt.figure()
plt.plot(error_prog)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

plt.show()
print('\nTest reuslt')
data_test = [[10, 3]]
for item in data_test:
    print(item, '-->', perceptron.sim([item])[0])