# Library for SA algorithm
# Comprised by Danyang Zhang @THU

import alg_lib
import numpy as np

def exp_temp_reducer(lambd):
    """
    return a exponential temperature reducer with parameter `lambd`
    """

    return lambda step, curr_temp: curr_temp*lambd

def optm_obj_not_vary_for(step_thr):
    """
    This function returns a co-routine generator. When the detected optimized objective value does not vary for `step_thr` steps, the generator will return `True` to terminate the SA algorithm.

    step_thr - the SA algorithm is required to be terminated when the detected optimized objective value doesn't vary for al least `step_thr` steps

    return a co-routine generator

    Co-routine accepts:
    step - the current iteration step
    curr_obj - the current objective value
    curr_optimized_obj - the so far optimized objective value
    curr_temp - the current temperature

    Co-routine generates:
    whether the algorithm should be terminated
    """

    last_optm_obj = None
    nb_not_varying_step = 0
    while True:
        _, _, optm_obj, _ = (yield nb_not_varying_step>=step_thr)
        if optm_obj==last_optm_obj:
            nb_not_varying_step += 1
        else:
            nb_not_varying_step = 0
            last_optm_obj = optm_obj

def step_reaches(step_thr):
    """
    return a convergence indicator which make judgement with only accordance to the number of the iteration step
    """

    return lambda step, curr_obj, curr_optimized_obj, curr_temp: step>=step_thr

def temp_lower_than(temp_thr):
    """
    return a convergence indicator which make judgement with only accordance to the current temperature
    """

    return lambda step, curr_obj, curr_optimized_obj, curr_temp: curr_temp<temp_thr

def obj_stablized(diff_thr, step_thr):
    """
    This function returns a co-routine generator. When the different of sequential two objective values is less than `diff_thr` for consecutive `step_thr` times, the generator will return `True` to reduce the temperature.

    diff_thr - difference threshold
    step_thr - step threshold

    return a co-routine generator

    Co-routine accepts:
    step - the current iteration step
    curr_obj - the current objective value
    curr_optimized_obj - the so far optimized objective value
    curr_temp - the current temperature

    Co-routine generates:
    whether the sampling has got stable so that the temperature could be reduced
    """

    nb_little_diff_time = 0
    last_obj = None
    last_temp = None
    _, last_obj, _, last_temp = (yield)
    while True:
        _, curr_obj, _, curr_temp = (yield nb_little_diff_time>=step_thr)
        if last_temp!=curr_temp or abs(curr_obj-last_obj)>=diff_thr:
            nb_little_diff_time = 0
        else:
            nb_little_diff_time += 1
        last_obj = curr_obj
        last_temp = curr_temp

def is_sampling_thorough(nb_sample_thr):
    """
    This function returns a co-routine generator. When `nb_sample_thr` samples have been fetched under the underlying temperature, the generator will return `True` to reduce the temperature.

    diff_thr - difference threshold
    step_thr - step threshold

    return a co-routine generator

    Co-routine accepts:
    step - the current iteration step
    curr_obj - the current objective value
    curr_optimized_obj - the so far optimized objective value
    curr_temp - the current temperature

    Co-routine generates:
    whether the sampling has got stable so that the temperature could be reduced
    """

    nb_sample = 0
    last_temp = None
    _, _, _, last_temp = (yield)
    while True:
        _, _, _, curr_temp = (yield nb_sample>=nb_sample_thr)
        if last_temp==curr_temp:
            nb_sample += 1
        else:
            nb_sample = 0
        last_temp = curr_temp

def simulated_annealing(obj, x_range,
        init_solution, init_temp,
        generator, temp_reducer,
        converges, gets_stable_sampling, accepts=alg_lib.boltzmann):
    """
    obj - objective function, form:
        double obj(double_ndarray solution)
    x_range - ndarray of [xi_min, xi_max], shape: (dim, 2)
    init_solution - initial solution
    init_temp - initial temperature
    generator - generates the next solution according to the current solution, form:
        double_ndarray generator(double_ndarray curr_solution)
    temp_reducer - reduces the temperature according to the current step and the current temperature, form:
        double temp_reducer(int step, double curr_temp)
    converges - judge whether the algorithm has converged, form:
        boolean converges(int step, double curr_obj, double curr_optimized_obj, double curr_temp)
    gets_stable_sampling - judge whether the sampling has got stable and the temperature should be reduced, form:
        boolean gets_stable_sampling(int step, double curr_obj, double curr_optimized_obj, double curr_temp)
    accepts - judge whether the new solution is accepted, default to the boltzmann energy function, form:
        boolean accepts(double temp, double curr_obj, double new_obj)

    return:
    optimized x
    optimized f
    list of tuple (solution, objective)
    list of tuple (optimized solution, optimized objective)
    list of temperature
    """

    solution_range = []
    optm_solution_range = []
    objective_range = []
    optm_objective_range = []
    temp_range = []

    step = 0
    x = np.array(init_solution)
    f = obj(x)
    t = float(init_temp)

    optm_x = x
    optm_f = f

    solution_range.append(x)
    optm_solution_range.append(optm_x)
    objective_range.append(f)
    optm_objective_range.append(optm_f)
    temp_range.append(t)

    while not converges(step, f, optm_f, t):
        nx = generator(x)
        nf = obj(nx)
        if accepts(t, f, nf):
            x = nx
            f = nf
        if f<optm_f:
            optm_x = x
            optm_f = f

        solution_range.append(x)
        optm_solution_range.append(optm_x)
        objective_range.append(f)
        optm_objective_range.append(optm_f)
        temp_range.append(t)

        step += 1
        if gets_stable_sampling(step, f, optm_f, t):
            t = temp_reducer(step, t)

    return optm_x, optm_f, list(zip(solution_range, objective_range)), list(zip(optm_solution_range, optm_objective_range)), temp_range
