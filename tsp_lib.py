# Library for GA algorithm
# Comprised by Danyang Zhang @THU

import numpy as np
import random
import concurrent.futures as con

random.seed()

def eda_sample(adjacency_matrix, frequency_matrix, population_size, nb_optm_individual):
    """
    adjacency_matrix - ndarray of integer or double
    frequency_matrix - ndarray of integer
    population_size - integer
    
    return:
    city series - ndarray
    objective values - ndarray
    new frequency matrix - ndarray, shape: (nb_city, nb_city)
    """

    nb_city = len(adjacency_matrix)
    samples = np.zeros((population_size, nb_city), dtype=np.uint8)
    objs = np.zeros((population_size,))

    def sample():
        curr_frequency_matrix = np.array(frequency_matrix)
        city = 0
        seq = np.zeros((nb_city,), dtype=np.uint8)
        obj = 0
        curr_frequency_matrix[:, city] = 0
        for j in range(1, nb_city):
            integral = np.cumsum(curr_frequency_matrix[city, :])
            if integral[-1]!=0.:
                rand_var = np.random.randint(1, integral[-1]+1)
                next_city = np.searchsorted(integral, rand_var)
            else:
                remaining_city = list(set(range(nb_city)) - set(seq))
                ind = random.randrange(len(remaining_city))
                next_city = remaining_city[ind]

            seq[j] = next_city
            obj += adjacency_matrix[city, next_city]

            city = next_city
            curr_frequency_matrix[:, city] = 0
        obj += adjacency_matrix[city, 0]
        return seq, obj

    max_workers = 8
    thread_pool = con.ThreadPoolExecutor(max_workers)
    future_pool = []
    for i in range(population_size):
        future_pool.append(thread_pool.submit(sample))
    for i, f in enumerate(future_pool):
        seq, obj = f.result()
        samples[i, :] = seq
        objs[i] = obj

    new_frequency_matrix = np.zeros((nb_city, nb_city))
    sorter = np.argsort(objs)[:nb_optm_individual]
    for i in sorter:
        for j in range(nb_city-1):
            new_frequency_matrix[samples[i][j], samples[i][j+1]] += 1
        new_frequency_matrix[samples[i][nb_city-1], samples[i][0]] += 1

    return samples, objs, new_frequency_matrix

def estimation_of_distribution_algorithm(adjacency_matrix, population_size, nb_optm_individual, converges):
    """
    adjacency_matrix - the adjacency matrix of the underlying graph, list or ndarray
    population_size - the size of the population, integer
    nb_optm_individual - the number of the optimal individuals in the population; is required to be less than `population_size`; integer
    converges - indicates whether the algorithm could be terminated, form:
        boolean converges(int step, number curr_obj, number curr_optimized_obj)

    return:
    optimized city sequence, list
    optimized objective value
    the final frequency matrix
    list of tuple (city sequences: ndarray, objective values: ndarray)
    list of tuple (optimized city sequence: ndarray, optimized objective value)
    list of frequency matrices in the iteration
    """

    nb_city = len(adjacency_matrix)
    adjacency_matrix = np.array(adjacency_matrix)

    frequency_matrix = np.ones((nb_city, nb_city))
    diagonal_mask = np.arange(nb_city)
    frequency_matrix[diagonal_mask, diagonal_mask] = 0

    solution_range = []
    optm_solution_range = []
    frequency_matrix_range = []

    step = 0
    frequency_matrix_range.append(frequency_matrix)
    city_seqs, objs, frequency_matrix = eda_sample(adjacency_matrix, frequency_matrix, population_size, nb_optm_individual)
    ind = np.argmin(objs)
    optm_city_seq = city_seqs[ind]
    optm_obj = objs[ind]
    solution_range.append((city_seqs, objs))
    optm_solution_range.append((optm_city_seq, optm_obj))
    while not converges(step, objs, optm_obj):
        step += 1
        frequency_matrix_range.append(frequency_matrix)
        city_seqs, objs, frequency_matrix = eda_sample(adjacency_matrix, frequency_matrix, population_size, nb_optm_individual)
        ind = np.argmin(np.array(objs))
        if objs[ind]<optm_obj:
            optm_city_seq = city_seqs[ind]
            optm_obj = objs[ind]

        solution_range.append((city_seqs, objs))
        optm_solution_range.append((optm_city_seq, optm_obj))

        if step%20==0:
            print("{:d}: {:.4f}".format(step, optm_obj))
    frequency_matrix_range.append(frequency_matrix)

    return optm_city_seq, optm_obj, frequency_matrix, solution_range, optm_solution_range, frequency_matrix_range
