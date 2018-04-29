#!/bin/env python

from cuter_util import *
from solver_util import makeA, makeB, makeC, makeBasis
from simplex import Simplex
from debug_utils import pause

class DustParam:
    """
    Store all the dust non linear solver parameters
    """

    def __init__(self, init_rho=1, init_omega=1e-2, max_iter=200, max_sub_iter=2000, beta_opt=0.7, beta_fea=0.1,
                 theta=0.9, init_delta=1, omega_shrink=0.7, eps_opt=1e-4, eps_violation=1e-5,
                 sub_verbose=False, rescale=True):
        self.init_rho = init_rho
        self.init_omega = init_omega
        self.max_iter = max_iter
        self.max_sub_iter = max_sub_iter
        self.beta_opt = beta_opt
        self.beta_fea = beta_fea
        self.theta = theta
        self.init_delta = init_delta
        self.omega_shrink = omega_shrink
        self.eps_opt = eps_opt
        self.eps_violation = eps_violation
        self.sub_verbose = sub_verbose
        self.rescale = rescale


def v_x(c, adjusted_equatn):
    """
    Calcuate v_x as defined in the paper which is the 1 norm of constraint violation of `c` vector
    :param c: constraint value or linearized constraint value vector
    :param adjusted_equatn: boolean vector indicating if it is equation constraint
    :return:
    """
    if np.any(np.isnan(c)):
        return np.nan
    equality_violation = np.sum(np.abs(c[adjusted_equatn]))
    inequality_violation = np.sum(c[np.logical_and(np.logical_not(adjusted_equatn), (c > 0).flatten())])

    return equality_violation + inequality_violation

def linearModelPenalty(A, b, g, rho, d, adjusted_equatn):
    """
    Calculate the l(d, rho; x) defined in the paper which is the linearized model of penalty function
    rho*f(x) + dist(c(x) | C)
    :param A: Jacobian of constraints
    :param b: c(x) constraint function value
    :param g: gradient of objective
    :param rho: penalty parameter
    :param d: current direction
    :param adjusted_equatn: boolean vector indicating if it is equation constraint
    :return: l(d, rho; x)
    """

    c = A.dot(d) + b
    linear_model = g.T.dot(d) * rho + v_x(c, adjusted_equatn)
    return linear_model[0, 0]

def getPrimalObject(primal_var, g, rho, equatn):
    # Line 757, formula 4.1
    return primal_var.dot(makeC(g*rho, equatn));
def getDualObject(A, g, rho, b, dual_var, delta):
    # Line 763, formula 4.2
    return  \
        (b.T.dot(dual_var) -  \
        delta * np.sum(np.abs(rho * g.T + dual_var.T.dot(A))))[0]
def getRatioC(A, b, dual_var, primal_var, equatn, l_0, omega):
    # line 227: r_c = 1 - sqrt(X/l_0)
    # line 221: X = sum((1-dual(i)) * v(<a, d> + b)) + sum((1+dual(j)) * v(<a, d> + b))
    #           where i in E+(d), I+(d), j in I-(d) 
    X = 0
    m, n = A.shape
    for i in range(m):
        x_new = A[i, :].dot(primal_var) + b[i, 0]
        if x_new > 0:
            X += (1-dual_var[i]) * x_new
        elif x_new < 0 and equatn[i] == True:            
            X += (1+dual_var[i]) * np.abs(x_new)
    return 1-np.sqrt(X / (l_0 + 1e-8))
def getRatio(A, b, g, rho, primal_var, dual_var, delta, equatn, l_0):
    # Line 199, formula 2.16.
    # When rho is set to 0, it calculates ratio_fea, or it calculates ratio_obj
    primal_obj = getPrimalObject(primal_var, g, rho, equatn)
    dual_obj = getDualObject(A, g, rho, b, dual_var, delta)
    
    if rho == 0:
        # Take the positive part
        dual_obj = max(0, dual_obj)

    up = l_0 - primal_obj
    down = (l_0 - dual_obj) + 1e-5
    return up/down
def l0(b, equatn):
    # Line 201
    b = b.reshape(1, -1)[0]
    return np.sum(np.abs(b[equatn == True])) + np.sum(b[np.logical_and(equatn == False, b>0)])

def getLinearSearchDirection(A, b, g, rho, delta, cuter, dust_param, omega):
    equatn = cuter.setup_args_dict['adjusted_equatn']
    
    m, n = A.shape
    c_, A_, b_, basis_ = makeC(g*rho, equatn), makeA(A), makeB(b, delta, n), makeBasis(b, n)
    
    # Construct a simplex problem instance.
    linsov = Simplex(c_, A_, b_, basis_)

    beta_fea = dust_param.beta_fea 
    beta_opt = dust_param.beta_opt
    theta = dust_param.theta
    
    dual_var = np.zeros(m)
    primal_var = np.zeros(n)
    ratio_opt = 0;
    ratio_fea = 0;
    ratio_c = 0

    l_0 = l0(b, equatn)
    iter_cnt = 0
    while not linsov.isOptimal():
        iter_cnt += 1;
        if iter_cnt > dust_param.max_iter:
        	break;
        # update the basis.
        linsov.updateBasis()

        # primal has size 4*n + 2*m
        # but we are only interested with the first 2n, as they are plus and minus of d.
        primal = linsov.getPrimalVar()
        # primal_var = d+ - d-
        primal_var = primal[0:n] - primal[n:2*n]

        # dual_var also has size m+2n,
        # but we are only interested with the first m.
        dual = -linsov.getDualVar()
        dual_var = dual[0, 0:m]
        nu_var = -linsov.getNuVar(makeC(g*0, equatn))

        # Update ratios.
        ratio_fea = getRatio(A, b, g, 0, primal, nu_var[0:m], delta, equatn, l_0)
        ratio_opt = getRatio(A, b, g, rho, primal, dual_var, delta, equatn, l_0)
        ratio_c = getRatioC(A, b, dual_var, primal_var, equatn, l_0, omega)

        # Debugging.
        #pause("ratio_fea", ratio_fea, "ratio_opt", ratio_opt, "ratio_c", ratio_c,"primal_var", primal_var, "dual_var", dual_var)
        # Update rho if needed.
        if ratio_c >= beta_fea and ratio_opt >= beta_opt and ratio_fea >= beta_fea:
        # Should all satisfies, break.
        # We don't do it now.
            break
        elif ratio_c >= beta_fea and ratio_opt >= beta_opt:
            rho *= theta
            linsov.resetC(makeC(g*rho, equatn))
    return primal_var.reshape((n, 1)), dual_var.reshape((m, 1)), rho, ratio_c, ratio_opt, ratio_fea, iter_cnt

def get_f_g_A_b_violation(x_k, cuter, dust_param):
    """
    Calculate objective function, gradient, constraint function and constraint Jacobian
    :param x_k: current iteration x
    :param cuter instance
    :param dust_param: dust param instance
    :return:
    """
    f, g = cuter.get_f_g(x_k, grad_flag=True, rescale=dust_param.rescale)
    b, A = cuter.get_constr_f_g(x_k, grad_flag=True, rescale=dust_param.rescale)
    violation = v_x(b, cuter.setup_args_dict['adjusted_equatn'])

    return f, g, b, A, violation

def linearSolveTrustRegion(cuter, dust_param, logger):
    """
    Non linear solver for cuter problems
    :param cuter instance
    :param dust_param: dust parameter class instance
    :param logger: logger instance to store log information
    :return:
        status:
                -1 - max iteration reached
                1 - solve the problem to optimality
    """
    def get_KKT(A, b, g, eta, rho):
        """
        Calcuate KKT error
        :param A: Jacobian of constraints
        :param b: c(x) constraint function value
        :param g: gradient of objective
        :param eta: multiplier in canonical form dual problem
        :param rho: penalty paramter rho
        :return: kkt error
        """

        err_grad = np.max(np.abs(A.T.dot(eta/rho) + g))    
        err_complement = np.max(np.abs(eta * b))
        #print A.T.dot(eta/rho) + g
        #print g * rho
        return max(err_grad, err_complement)

    setup_args_dict = cuter.setup_args_dict
    x_0 = setup_args_dict['x']
    num_var = setup_args_dict['n'][0]
    beta_l = 0.6 * dust_param.beta_opt * (1 - dust_param.beta_fea)
    adjusted_equatn = cuter.setup_args_dict['adjusted_equatn']
    zero_d = np.zeros(x_0.shape)
    i, status = 0, -1
    x_k = x_0.copy()

    logger.info('-' * 200)
    logger.info(
        '''{0:4s} | {1:13s} | {2:12s} | {3:12s} | {4:12s} | {5:12s} | {6:12s} | {7:12s} | {8:12s} | {9:12s} | {10:6s} | {11:12s} | {12:12s} | {13:12s}'''.format(
            'Itr', 'KKT', 'Step_size', 'Violation', 'Rho', 'Objective', 'Ratio_C', 'Ratio_Fea', 'Ratio_Opt', 'Omega',
            'SubItr', 'Delta_L', 'Merit', "||d||"))

    f, g, b, A, violation = get_f_g_A_b_violation(x_k, cuter, dust_param)
    print np.sum(np.abs(g))
    m, n = A.shape
    rho = dust_param.init_rho
    omega = dust_param.init_omega
    max_iter = dust_param.max_iter
    rescale = dust_param.rescale

    # Only required for SQP
    # Initialize dual variables
    #dual_var = initialize_dual_var(adjusted_equatn, b)
    #lam = initialize_dual_var(adjusted_equatn, b)
    kkt_error_k = get_KKT(A, b, g, np.zeros((m, 1)), rho)

    all_rhos, all_kkt_erros, all_violations, all_fs, all_sub_iter = \
        [dust_param.init_rho], [kkt_error_k], [violation], [f], []

    fn_eval_cnt = 0

    logger.info(
        '''{0:4d} |  {1:+.5e} | {2:+.5e} | {3:+.5e} | {4:+.5e} | {5:+.5e} | {6:+.5e} | {7:+.5e} | {8:+.5e} | {9:+.5e} | {10:6d} | {11:+.5e} | {12:+.5e} | {13:+.5e}''' \
            .format(i, kkt_error_k, -1, violation, rho, f, -1, -1, -1, omega, -1, -1, rho * f + violation, -1))


    step_size = -1.0
    delta = 1; 

    while i < max_iter:

        # DUST / PSST / Subproblem here.
        #dual_var, d_k, lam, rho, ratio_complementary, ratio_opt, ratio_fea, sub_iter, H_rho = \
        #    get_search_direction(x_k, dual_var, lam, rho, omega, A, b, g, cuter, dust_param)
        d_k, dual_var, rho, ratio_complementary, ratio_opt, ratio_fea, sub_iter = \
            getLinearSearchDirection(A, b, g, rho, delta, cuter, dust_param, omega)
        # 2.3
        l_0_rho_x_k = linearModelPenalty(A, b, g, rho, zero_d, adjusted_equatn)
        l_d_rho_x_k = linearModelPenalty(A, b, g, rho, d_k, adjusted_equatn)
        delta_linearized_model = l_0_rho_x_k - l_d_rho_x_k
        # 2.2
        l_0_0_x_k = linearModelPenalty(A, b, g, 0, zero_d, adjusted_equatn)
        l_d_0_x_k = linearModelPenalty(A, b, g, 0, d_k, adjusted_equatn)
        delta_linearized_model_0 = l_0_0_x_k - l_d_0_x_k
        
        # Don't know what kke error is yet.
        kkt_error_k = get_KKT(A, b, g, dual_var, rho)
        # ratio_opt: 3.6. It's actually r_v in paper.
        x_k += d_k
        # PSST
        if delta_linearized_model_0 > 0 and \
                delta_linearized_model + omega < beta_l * (delta_linearized_model_0 + omega):
            # TODO: Change this update, as we are using linear.
            rho = (1 - beta_l) * (delta_linearized_model_0 + omega) / (g.T.dot(d_k))[0, 0]

        f, g, b, A, violation = get_f_g_A_b_violation(x_k, cuter, dust_param)
        kkt_error_k = get_KKT(A, b, g, dual_var, rho)
        omega *= dust_param.omega_shrink

        # Store iteration information
        all_rhos.append(rho)
        all_violations.append(violation)
        all_fs.append(f)
        all_kkt_erros.append(kkt_error_k)
        all_sub_iter.append(sub_iter)

        logger.info(
            '''{0:4d} |  {1:+.5e} | {2:+.5e} | {3:+.5e} | {4:+.5e} | {5:+.5e} | {6:+.5e} | {7:+.5e} | {8:+.5e} | {9:+.5e} | {10:6d} | {11:+.5e} | {12:+.5e} | {13:+.5e}''' \
                .format(i, kkt_error_k, step_size, violation, rho, f, ratio_complementary, ratio_fea, ratio_opt, omega,
                        sub_iter, delta_linearized_model, rho * f + violation, np.linalg.norm(d_k, 2)))

        if kkt_error_k < dust_param.eps_opt and violation < dust_param.eps_violation:
            status = 1
            break
        i += 1
    logger.info('-' * 200)

    if rescale:
        f = f / setup_args_dict['obj_scale']

    return {'x': x_k, 'dual_var': dual_var, 'rho': rho, 'status': status, 'obj_f': f, 'x0': setup_args_dict['x'],
            'kkt_error': kkt_error_k, 'iter_num': i, 'constraint_violation': violation, 'rhos': all_rhos,
            'violations': all_violations, 'fs': all_fs, 'subiters': all_sub_iter, 'kkt_erros': all_kkt_erros,
            'fn_eval_cnt': fn_eval_cnt, 'num_var': num_var, 'num_constr': dual_var.shape[0]}
