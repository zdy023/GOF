# Several general functions
# Comprised by Danyang Zhang @THU

import random
import math
import numpy as np

random.seed()

# implement SA and PSO for continuous optimazation
# implement GA and EDA for discrete optimazation

def random_along_each_dim_generator(x_range, shift_coef):
    """
    return a generator which generates the new solution according to the formula:
        xi' = xi + ksi*shift_coef
        where ksi ~ U(-1, 1)
    """

    def generator(curr_solution):
        new_solution = curr_solution + shift_coef*np.random.uniform(-1, 1, size=curr_solution.shape)
        new_solution = np.clip(new_solution, x_range[:, 0], x_range[:, 1])
        return new_solution
    return generator

def random_along_orientation_generator(x_range, y_range, shift_coef):
    """
    return a generator which generates the new solution according to the formula:
        x' = x + stride*(cos(theta), sin(theta))
        there theta is a random angle, and stride is random distance
    """

    def generator(curr_solution):
        angle = random.uniform(0, 2*math.pi)
        dx = math.cos(angle)
        dy = math.sin(angle)
        stride = shift_coef*random.random()
        nx = curr_solution[0] + stride*dx
        ny = curr_solution[1] + stride*dy
        nx = max(x_range[0], nx)
        nx = min(x_range[1], nx)
        ny = max(y_range[0], ny)
        ny = min(y_range[1], ny)
        return np.array([nx, ny])
    return generator

def complt_random_generator(x_range):
    return lambda curr_solution: np.random.uniform(x_range[:, 0], x_range[:, 1], size=curr_solution.shape)

def exp_reducer(period, lambd):
    """
    return a exponential reducer which reduces the specific parameter with ratio `lambd` each `period` steps
    """

    return lambda step, curr_temp: curr_temp*lambd if step%period==0 else curr_temp

def linear_reducer(init_value, grad):
    """
    return a linear reducer which reduces the parameter according to the formula:
        para = max{init_value - grad*step, 0}
    """

    return lambda step, curr_para: max(0., init_value - grad*step)

def boltzmann(temp, curr_obj, new_obj):
    """
    Boltzmann energy function

    temp - current temperature
    curr_obj - current objective
    new_obj - new objective

    return whether the new solution should be accepted
    """

    if new_obj < curr_obj:
        return True
    thre = math.exp((curr_obj-new_obj)/temp)
    tmp = random.random()
    return tmp<thre

def optm_obj_not_vary_for(step_thr):
    """
    This function returns a co-routine generator. When the detected optimized objective value does not vary for `step_thr` steps, the generator will return `True`.

    step_thr - `True` is returned when the detected optimized objective value doesn't vary for al least `step_thr` steps

    return a co-routine generator

    Co-routine accepts:
    step - the current iteration step
    curr_obj - the current objective value
    curr_optimized_obj - the so far optimized objective value

    Co-routine generates a boolean
    """

    last_optm_obj = None
    nb_not_varying_step = 0
    while True:
        _, _, optm_obj = (yield nb_not_varying_step>=step_thr)
        if optm_obj==last_optm_obj:
            nb_not_varying_step += 1
        else:
            nb_not_varying_step = 0
            last_optm_obj = optm_obj

def step_reaches(step_thr):
    """
    return a convergence indicator which make judgement with only accordance to the number of the iteration step
    """

    return lambda step, curr_obj, curr_optimized_obj: step>=step_thr

def para_lower_than(threshold):
    """
    return a convergence indicator which returns `True` then the given parameter is less than `threshold`
    """

    return lambda step, curr_obj, curr_optimized_obj, extra_para: extra_para<threshold
