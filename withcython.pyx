import numpy as np
cimport numpy as np
import cython
from libc.math cimport exp

np.import_array()

cpdef np.ndarray[np.float64_t, ndim=2] g(np.ndarray[np.float64_t, ndim=2] x):
    return 1.0 / (1.0 + np.exp(-x))

cpdef np.ndarray[np.float64_t, ndim=2] grad_g(np.ndarray[np.float64_t, ndim=2] x):
    cdef np.ndarray[np.float64_t, ndim=2] gx = g(x)
    return gx * (1.0 - gx)

cdef tuple reshape(np.ndarray[np.float64_t, ndim=1] theta, unsigned int input_layer_size, unsigned int hidden_layer_size, unsigned int num_labels):
    cdef unsigned int ncut = hidden_layer_size * (input_layer_size + 1)
    cdef np.ndarray[np.float64_t, ndim=2] Theta1 = theta[:ncut].reshape((hidden_layer_size, input_layer_size + 1), order='C')
    cdef np.ndarray[np.float64_t, ndim=2] Theta2 = theta[ncut:].reshape((num_labels, hidden_layer_size + 1), order='C')
    return Theta1, Theta2

@cython.boundscheck(False)  # deactivate bounds checking
cpdef np.ndarray[np.float64_t, ndim=1] gradient(np.ndarray[np.float64_t, ndim=1] theta,
                                                unsigned int input_layer_size,
                                                unsigned int hidden_layer_size,
                                                unsigned int num_labels,
                                                np.ndarray[np.float64_t, ndim=2] X,
                                                np.ndarray y,
                                                double lmbda):

    cdef np.ndarray[np.float64_t, ndim=2] Theta1, Theta2
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)

    cdef unsigned int m = y.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    cdef np.ndarray[np.float64_t, ndim=2] Delta2 = np.zeros((num_labels, hidden_layer_size + 1))

    cdef np.ndarray[np.float64_t, ndim=2] a1 = X.T
    a1 = np.vstack((np.ones((1, X.shape[0])), a1))
    cdef np.ndarray[np.float64_t, ndim=2] z2 = np.dot(Theta1, a1)
    cdef np.ndarray[np.float64_t, ndim=2] a2 = g(z2)
    a2 = np.vstack((np.ones((1, X.shape[0])), a2))
    cdef np.ndarray[np.float64_t, ndim=2] a3 = g(np.dot(Theta2, a2))  

    cdef np.ndarray[np.float64_t, ndim=2] y_k = np.zeros((num_labels, X.shape[0]))
    for t in range(m):
        y_k[y[t, 0].astype(int), t] = 1.0

    cdef np.ndarray[np.float64_t, ndim=2] delta3 = a3 - y_k 
    Delta2 += np.dot(delta3, a2.T) 

    cdef np.ndarray[np.float64_t, ndim=2] delta2 = np.dot(Theta2[:, 1:].T, delta3) * grad_g(z2) 
    Delta1 += np.dot(delta2, a1.T)   

    cdef np.ndarray[np.float64_t, ndim=2] Theta1_grad = Delta1 / m
    cdef np.ndarray[np.float64_t, ndim=2] Theta2_grad = Delta2 / m

    Theta1_grad[:, 1:] += (lmbda / m) * Theta1[:, 1:]  
    Theta2_grad[:, 1:] += (lmbda / m) * Theta2[:, 1:]

    cdef np.ndarray[np.float64_t, ndim=1] grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return grad