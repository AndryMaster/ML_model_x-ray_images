import numpy as np


def get_gradient(func, x, dx=1e-9):
    return ((func(x + dx) - func(x)) / dx).real


def gradient_descent(func, start_point: int, gamma, epsilon, steps):
    x = start_point
    x_history = [x]

    run = not steps
    iteration = 0
    while run or iteration < steps:
        grad = get_gradient(func, x)
        x_next = x - gamma * grad
        diff_loss = abs(func(x_next) - func(x))

        x_history.append(x_next)
        # print(f"Iteration {iteration + 1}:    {diff_loss :.6f}\n{grad= :.3f}\t{x_next= :.3f}")

        if diff_loss < epsilon:
            run = False
        x = x_next
        iteration += 1

    return np.reshape(np.round(x_history, 3), newshape=(-1, 1))


f = lambda x: (x + 5)**2 - 7
print(gradient_descent(f, 70, 0.09, 1e-9, steps=20))

# f = lambda x: x*25 + (x-3)**2.4 - 84 - (x+89)**.5
# print(gradient_descent(f, 24000, 0.01, 1e-6, steps=500))
