import numpy as np


class Variable:

        def __init__(self, value: float = 0.):
                self.value = value
                self.grad = np.zeros_like(value)

        def backward(self):

                self.backward = int
                self.back()

                for variable in self.variables:
                        variable.backward()
