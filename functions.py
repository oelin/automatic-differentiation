from variable import Variable


def add(x, y):

        z = Variable(x.value + y.value)
        z.variables = [x, y]

        def back():
                x.grad += z.grad
                y.grad += z.grad

        z.back = back
        return z


def sub(x, y):

        z = Variable(x.value - y.value)
        z.variables = [x, y]

        def back():
                x.grad += z.grad
                y.grad -= z.grad

        z.back = back
        return z


def multiply(x, y):

        z = Variable(x.value * y.value)
        z.variables = [x, y]

        def back():
                x.grad += z.grad * y.value
                y.grad += z.grad * x.value

        z.back = back
        return z


def sum(x):
        z = Variable(np.sum(x.value))
        z.variables = [x]

        def back():
                x.grad += z.grad.item()

        z.back = back
        return z


def tensor_product(x, y):

        z = Variable(x.value @ y.value)
        z.variables = [x, y]

        def back():
                x.grad += z.grad @ y.value.T
                y.grad += x.value.T @ z.grad

        z.back = back
        return z
