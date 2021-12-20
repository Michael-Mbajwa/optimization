# Import all packages to be used
import sympy as sym
import numpy as np
from numpy.linalg import inv, norm
import copy

# Line code 7 - 21 has to be computed manually for different functions
# Let's consider the function
# f: (x1, x2, x3, x4) |-> (x1 + 10x2)^4 +5(x3 - x4)^4 + (x2 - 2x3)^4 + 10(x1 - x4)^4

# I declare the variables
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
x3 = sym.Symbol('x3')
x4 = sym.Symbol('x4')

# Variables are stored in a list
variables = [x1, x2, x3, x4]

# I expand the given function f(x1, x2, x3, x4)
function_1 = sym.expand((x1 + 10*x2)**2 + 5*(x3 - x4)**4 + (x2 - 2*x3)**4 + 10*(x1-x4)**4)


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


def hessian(func, points):
    """
    Function calculates the hessian matrix (a square matrix of second order order differentials).
    :param func:
    :param points:
    :return: A square matrix of second order order differentials. Assuming n = number of variables.
    The dimension of this square matrix will be n * n.
    """
    grad = gradient(func, points)  # First step is to calculate the gradient of the function using the gradient function
    length = len(points)  # Determine the number of variables
    # I now create an n*n matrix to be used to store the values of the hessian matrix
    hess_mat = np.array([None] * (length**2)).reshape(length, length)  # The n*n matrix
    for i in range(length):
        partial_derivative = grad[i]  # Loop through each value of the gradient
        for j in range(length):
            # Loop through each variable
            hess_mat[i][j] = sym.diff(partial_derivative, points[j])  # Calculate second order differential and store

    return hess_mat


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


def hessian_subs(hess, x):
    """
    This function substitutes real-number values into the hessian matrix.
    :param hess: the hessian matrix
    :param x: a list containing all values to be substituted.
    :return: returns a hessian matrix with substituted values.
    """
    holder = copy.deepcopy(hess)  # Deep copy the gradient
    a, b, c, d = x  # I unpack the items of the list containing the values
    for i in range(len(x)):  # Loop through each item in the matrix
        for j in range(len(x)):
            if type(holder[i][j]) == int:  # if the item is already an integer, skip it
                pass
            else:
                holder[i][j] = holder[i][j].subs([(x1, a), (x2, b), (x3, c), (x4, d)])
    holder = holder.astype(float)  # Convert the values to slope so I can use the norm function later
    return holder


def newton(f, var, start, eps):
    """
    The implementation of the newton algorithm.
    :param f: function
    :param var: variables for the function
    :param start: a list containing start values
    :param eps: the precision
    :return: returns three values as a string. Returns xk, f(xk) and number of iterations.
    """
    xk = np.array(start)  # Initialize starting point as a numpy array
    k = 0  # Initialize iteration

    diff = gradient(f, var)  # Calculates the gradient
    hessian_mat = hessian(f, var)  # Calculates the hessian matrix

    while norm(grad_subs(diff, xk)) > eps:  # If the norm of the gradient in terms of xk > greater than precision
        gradient_vec = grad_subs(diff, xk)  # Substitute values of xk into gradient
        hessian_matrix = hessian_subs(hessian_mat, xk)  # Substitute values of xk into hessian matrix
        xk = xk - np.dot(inv(hessian_matrix), gradient_vec)  # xk+1 := xk - inv([Hf (xk)]) * ∇f(xk)
        k += 1

    fk = f.subs([(x1, xk[0]), (x2, xk[1]), (x3, xk[2]), (x4, xk[3])])  # Calculates the value of f(xk)

    final_result = "xk: {}\nf(xk): {}\nIterations: {}".format(xk, fk, k)  # Final output to be returned
    return final_result


if __name__ == "__main__":
    print(newton(function_1, variables, [3, -1, 0, 1], 0.01))

    # Returns
    # xk: [0.08670765 - 0.00867076  0.00867076  0.03468306]
    # f(xk): 0.0000760016046916214
    # Iterations: 9
