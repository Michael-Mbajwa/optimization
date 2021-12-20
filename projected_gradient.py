# Import all packages to be used
import sympy as sym
import numpy as np
from numpy.linalg import inv, norm
import copy

# Line code 7 - 25 has to be computed manually for different functions
# Let's consider the function
# f(x) = (x1 + x2)^2 + (x3 + x4)^4 + 3(x1 -2)^2 + (2x3 +1)^2 + 2(x4 -0.5)^2 + 1

# I declare the variables
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
x3 = sym.Symbol('x3')
x4 = sym.Symbol('x4')

# Variables are stored in a list
variables = [x1, x2, x3, x4]

# Define given function on sympy
function_2 = sym.expand((x1 + x2)**2 + (x3 + x4)**4 + 3*(x1 - 2)**2 + (2*x3+1)**2 + 2*(x4 - 0.5)**2 + 1)

# Define projection operator for R**4
vector = np.array([x1-x2, -x1+x2, x3-x4, -x3+x4])
Pf = 0.5*vector


def gradient(func, points):
    """
    This code returns the gradient of the function provided i.e ∇func. It differentiates the function with respect
    to the all its variables. I use the sympy package for differentiation.
    :param func: The function to be differentiated. e.g x**2-y
    :param points: The various variables contained in the function. e.g x and y. x1, x2, x3 and x4.
    :return: returns a vector in form of a numpy array containing the differentiated function with respect to each
    variable was provided. If  f=x**2-y and points=(x1, y) is provided, the code returns [2x, -1].
    """
    length = len(points)  # Get the number of variables. This determines the length of the gradient matrix.
    grad = np.array([None] * length)  # Create the vector that stores all differentials.
    for i in range(length):
        grad[i] = sym.diff(func, points[i])  # Loop through each variable and differentiate function with respect to it

    return grad


def grad_subs(grad, x):
    """
    This function substitutes real-number values into the gradient.
    :param grad: the result from a gradient vector. In the form of an array.
    :param x: a list containing all values to be substituted.
    :return: a list
    """
    a, b, c, d = x  # I unpack the items of the list containing the values
    holder = copy.deepcopy(grad)  # Deep copy the gradient
    for i in range(len(holder)):  # Loop through each value of the gradient and substitute the values.
        holder[i] = holder[i].subs([(x1, a), (x2, b), (x3, c), (x4, d)])
    holder = holder.astype(float)
    # I cannot calculate norm in python with integer so I convert the array values to float
    return holder


def projector_subs(proj, val):
    """
    Function for substituting real number values in projector operator
    :param proj: the projector operator in form of a numpy array
    :param val: the values to be substituted in order
    :return: returns projector operator vector with susbtituted values
    """
    holder = copy.deepcopy(proj)  # Deep copy so we don't edit the original projector operator
    a, b, c, d = val  # Unpacks the values provided
    for i in range(len(proj)):
        holder[i] = holder[i].subs([(x1, a), (x2, b), (x3, c), (x4, d)])
    return holder


def proj_grad_desc(func, pf, var, start, eps, s=0.2):
    xk = np.array(start)  # Initialize starting point as a numpy array
    k = 0  # Initialize iteration
    diff = gradient(func, var)  # Calculates the gradient

    while norm(grad_subs(diff, xk)) > eps:  # If the norm of the gradient in terms of xk > greater than precision
        gradient_vec = grad_subs(diff, xk)  # Substitute values of xk into gradient
        yk = xk - s*gradient_vec

        xk = projector_subs(pf, yk)  # Substitute values of yk into projector operator to get xk
        k += 1  # Iterator increment

    fk = func.subs([(x1, xk[0]), (x2, xk[1]), (x3, xk[2]), (x4, xk[3])])  # Calculates the value of f(xk)

    final_result = "xk: {}\nf(xk): {}\nIterations: {}".format(xk, fk, k)  # Final output to be returned
    return final_result


if __name__ == "__main__":
    print(proj_grad_desc(function_2, Pf, variables, [1, -1, 1, -1], 0.01))
    # Returns
    # xk: [1.99836160000000 - 1.99836160000000 - 0.500019200000000 0.500019200000000]
    # f(xk): 1.00000805527552
    # Iterations: 7
