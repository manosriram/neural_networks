import numpy
import scipy.special

def activation_function(inp):
    return scipy.special.expit(inp)

wih = numpy.random.rand(3, 3) - 0.5
who = numpy.random.rand(3, 3) - 0.5

inputs = numpy.array([0.1, 0.9, 0.5], ndmin=2).T
targets = numpy.array([0, 0, 1], ndmin=2).T


input_hidden = numpy.dot(wih, numpy.random.rand(3))
output_hidden = activation_function(input_hidden)

input_final = numpy.dot(who, output_hidden)
output_final = activation_function(input_final)


error_output = targets - output_final

error_hidden = numpy.dot(who.T, error_output)
print(error_hidden)
