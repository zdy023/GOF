# Library for PSO algorithm
# Comprised by Danyang Zhang @THU

import numpy as np

def particle_swarm_optimazation(obj, x_range,
        swarm_size,
        inertia_coef, personal_influence_coef, social_influence_coef,
        init_inertia_coef, init_personal_influence_coef, init_social_influence_coef,
        converges):
    """
    obj - objective function, form:
        double obj(double_ndarray x)
    x_range - range of the solution, shape: (dim, 2)
    swarm_size - the size of the swarm, integer
    inertia_coef - the inertia coefficient (w) in
        vi(t+1) = w(t)*vi(t) + phy1*(bi(t)-xi(t)) + phy2*(gbi(t)-xi(t))
        this parameter is expected as an executable, form:
        double inertia_coef(int step, double curr_inertia)
    personal_influence_coef - the personal influence coefficient, i.e. phy1 in the formula; this parameter is expected as an executable as well; form:
        double personal_influence_coef(int step, double curr_coef)
    social_influence_coef - the social influence coefficient, i.e. phy2 in the formula; executable as well; form:
        double social_influence_coef(int step, double curr_coef)
    init_inertia_coef - init inertia coefficient
    init_personal_influence_coef - init personal influence coefficient
    init_social_influence_coef - init social influence coefficient
    converges - the indicator of whether the algorithm should be terminated, form:
        boolean converges(int step, double curr_obj, double curr_optimized_obj)

    return:
    optimized x
    optimized f
    list of tuple (ndarray of solution, ndarray of objective)
    list of tuple (optimized solution, optimized objective)
    """

    solution_range = []
    optm_solution_range = []
    objective_range = []
    optm_objective_range = []

    step = 0
    xs = np.random.uniform(x_range[:, 0], x_range[:, 1], size=(swarm_size, len(x_range)))
    fs = obj(xs)
    vs = np.zeros(xs.shape)

    optm_xs = xs
    optm_fs = fs

    optm_ind = np.argmin(optm_fs)
    optm_x = optm_xs[optm_ind]
    optm_f = optm_fs[optm_ind]

    solution_range.append(xs)
    objective_range.append(fs)
    optm_solution_range.append(optm_x)
    optm_objective_range.append(optm_f)

    w = init_inertia_coef
    phy1 = init_personal_influence_coef
    phy2 = init_social_influence_coef
    while not converges(step, fs, optm_f):
        vs = w*vs + phy1*(optm_xs-xs) + phy2*(optm_x-xs)
        nxs = xs+vs
        nxs = np.clip(nxs, x_range[:, 0], x_range[:, 1])
        nfs = obj(nxs)

        accepts = nfs<fs
        xs = np.where(accepts[:, None], nxs, xs)
        fs = np.where(accepts, nfs, fs)

        accepts = fs<optm_fs
        optm_xs = np.where(accepts[:, None], xs, optm_xs)
        optm_fs = np.where(accepts, fs, optm_fs)

        optm_ind = np.argmin(optm_fs)
        optm_x = optm_xs[optm_ind]
        optm_f = optm_fs[optm_ind]

        solution_range.append(xs)
        objective_range.append(fs)
        optm_solution_range.append(optm_x)
        optm_objective_range.append(optm_f)

        step += 1
        w = inertia_coef(step, w)
        phy1 = personal_influence_coef(step, phy1)
        phy2 = social_influence_coef(step, phy2)

    return optm_x, optm_f, list(zip(solution_range, objective_range)), list(zip(optm_solution_range, optm_objective_range))
