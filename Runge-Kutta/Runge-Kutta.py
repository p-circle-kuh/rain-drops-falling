import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Initial Values
T_INITIAL = 0  # [s]
V_INITIAL = 0  # [m/s]

# Params for Phys.
k = 1  # [1/s]
g = 9.8  # [m/s]

# Params for Calc.
T_FINAL = 10  # [s]
T_STEP = 0.1  # [s]


def derivative(t, v):  # Solver needs the variable t
    return g - k * v


def plot(t, v):
    plt.figure(figsize=(8, 6))
    plt.plot(t, v, '-o')
    plt.hlines([g], T_INITIAL, T_FINAL, "red", linestyles='dashed')
    plt.xlim([T_INITIAL, T_FINAL])
    plt.xlabel('t [s]')
    plt.ylabel('v (downward) [m/s]')
    plt.title(f'Velocity of a falling raindrop (g={g} [m/s^2], k={k} [1/s])')
    plt.legend(['RK4', '9.8'], loc='lower right')
    plt.show()


def main():
    n_steps = (T_FINAL - T_INITIAL) / T_STEP
    t_eval = T_INITIAL + np.arange(0, n_steps) * T_STEP
    result = solve_ivp(derivative, [T_INITIAL, T_FINAL], [V_INITIAL], t_eval=t_eval)

    if not result.success:
        print('Failed to solve!')
        return

    plot(result.t, result.y[0])  # solve_ivp returns t and multi-dimensional y 


if __name__ == '__main__':
    main()
