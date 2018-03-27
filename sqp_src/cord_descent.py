#!/bin/env python
import numpy as np


def get_inverse_cholesky(H):
    """
    Compute the cholesky decomposition of inverse of H
    :param H: input positive definite matrix
    :return: L such that LL_H = H^{-1}
    """
    H_inv = np.linalg.inv(H)
    L = np.linalg.cholesky(H_inv)
    return L


def get_primal_x(A, g, rho, dual_var, L_x):
    """
    Get the primal x corresponding to current dual solution
    For optimality problem x = -H_rho^{-1} (rho*g + A^T eta)
    For feasibility problem z = -H_0^{-1}A^T lam
    where dual_var is either eta or lam based on rho value
    :param A: Jacobian of constraints at current x_0
    :param g: gradient of orginal NLP objective at x_0
    :param rho: penalty parameter rho*H_f + H_0
    :param dual_var: dual variable, eta or lam
    :param L_x: L_rho or L_0 which is the Cholesky component of H^{-1}_x
    :return: primal x
    """
    tmp = rho * g + A.T.dot(dual_var)
    x = -L_x.dot(L_x.T.dot(tmp))
    return x


def get_primal_obj(H_x, rho, g, A, b, x_k, equatn):
    """
    Get the primal objective at x_k
    :param H_x: Hessian of Lagrangian or constraints at x_0
    :param rho: penalty parameter rho*H_f + H_0
    :param g: gradient of orginal NLP objective
    :param A: Jacobian of constraints at current x_0
    :param b: constraints value at current x_0
    :param equatn: boolean vector indicating equation constraint
    :return: primal objective value of optimality problem or feasibility problem (rho = 0)
    """
    violation = A.dot(x_k) + b
    violation[np.logical_and(np.logical_not(equatn), (violation < 0).flatten()), :] = 0
    violation = np.abs(violation)
    obj = 0.5 * x_k.T.dot(H_x.dot(x_k)) + rho * g.T.dot(x_k) + np.sum(violation)
    return obj[0, 0]


def get_dual_obj(A, b, rho, g, L_x, dual_var):
    """
    Get the dual objective at dual_var
    :param A: Jacobian of constraints at current x_0
    :param b: constraints value at current x_0
    :param g: g is None for feasibility, otherwise for optimality problem
    :param L_x: L_0 or L_rho
    :param dual_var
    :return:
    """
    A_T_dual_var_g = A.T.dot(dual_var)
    if g is not None:
        A_T_dual_var_g += rho * g

    tla = L_x.T.dot(A_T_dual_var_g)
    obj = -0.5 * tla.T.dot(tla) + dual_var.T.dot(b)
    return obj[0, 0]


def update_rho(g, A, b, raw_H_0, raw_H_f, rho, dual_var, eig_add_on):
    """
    Compute Choleksy decomposition of H_rho, AL_rho, dual problem gradient, v_dual and diag_rho
    :param g: gradient of orginal NLP objective
    :param A: Jacobian of constraints
    :param b: constraints function value at current x_0
    :param raw_H_0: Feasibility problem hessian
    :param raw_H_f: Optimality problem hessian
    :param rho: Penalty parameter rho * H_f + H_0
    :param dual_var: dual variables
    :param eig_add_on: add-on small constant to eigenvalue
    :return:
    """
    H_rho = raw_H_0 + rho * raw_H_f
    H_rho = positive_definify(H_rho, eig_add_on)
    L_rho = get_inverse_cholesky(H_rho)

    A_L_rho = A.dot(L_rho)
    tmp_L_rho_T_g = L_rho.T.dot(g)
    dual_obj_grad = A_L_rho.dot(-rho * tmp_L_rho_T_g) + b

    v_dual = A_L_rho.T.dot(dual_var)
    diag_rho = np.sum(A_L_rho ** 2, 1).reshape((-1, 1))

    return H_rho, L_rho, A_L_rho, dual_obj_grad, v_dual, diag_rho


def get_chi_du_l_d(A, b, d_k, eta, equatn, num_constraints):
    """
    Compute chi(d, u) and l(d, 0) defined in DUST paper
    :param A: Jacobian of constraints at current x_0
    :param b: constraints value at current x_0
    :param d_k: current d_k for optimality problem
    :param eta: dual variable for optimality problem
    :param equatn: boolean vector indicating equation constraint
    :param num_constraints: number of total constraints
    :return: chi(d, u) and l(d, 0)
    """
    linearized_constraint_val = A.dot(d_k) + b
    complement = np.ones((num_constraints, 1))
    pos_flag = (linearized_constraint_val > 0).flatten()
    eq_pos_ind = np.logical_and(pos_flag, equatn)
    complement[eq_pos_ind, :] -= eta[eq_pos_ind, :]
    neg_flag = (linearized_constraint_val < 0).flatten()
    eq_neg_flag = np.logical_and(neg_flag, equatn)
    complement[eq_neg_flag, :] += eta[eq_neg_flag, :]
    linearized_constraint_val[np.logical_and(np.logical_not(equatn), neg_flag), :] = 0
    linearized_constraint_val = np.abs(linearized_constraint_val)
    inequality_n = np.logical_not(equatn)
    complement[inequality_n, :] -= eta[inequality_n, :]
    chi_d_u = complement.T.dot(linearized_constraint_val)[0, 0]
    l_d_0 = np.sum(linearized_constraint_val)
    return chi_d_u, l_d_0


def positive_definify(H, eig_add_on):
    """
    Convert H to be a positive definite matrix if it is not by making all negative eigenvalues to be `eig_add_on`
    :param H: Matrix to be positive definified
    :param eig_add_on: smalle positive constant
    :return: Positive definified matrix
    """
    eigval, U = np.linalg.eig(H)

    # When matrix condition number is huge, may run into numerical issue. some complex eig value may appear
    if np.any(np.iscomplex(eigval)):
        eigval = np.real(eigval)
        U = np.real(U)

    max_eig = np.max(eigval)
    target_cond = 1e6
    if max_eig < 0:
        return eig_add_on * np.identity(H.shape[0])
    elif max_eig / max(eig_add_on, np.min(eigval)) < target_cond:
        eigval[eigval < eig_add_on] = eig_add_on
        return U.dot(np.diag(eigval)).dot(U.T) + eig_add_on * np.identity(H.shape[0])
    else:
        eigval[eigval < eig_add_on] = eig_add_on
        pd_H = U.dot(np.diag(eigval)).dot(U.T)
        current_cond = max_eig / np.min(eigval)
        return (target_cond / current_cond) * pd_H + eig_add_on * np.identity(H.shape[0])


def cord_descent(raw_H_0, raw_H_f, rho, g, A, b, equatn, omega, beta_fea, beta_opt, theta, max_iter, eig_add_on,
                 eta=None, lam=None, verbose=True):
    """

    :param raw_H_0: Hessian of constrains without eigenvalue correction
    :param raw_H_f: Hessian of objective without eigenvalue correction
    :param rho: penalty parameter rho*H_f + H_0
    :param g: gradient of orginal NLP objective
    :param A: Jacobian of constraints at current x_0
    :param b: constraints value at current x_0
    :param equatn: boolean vector indicating equation constraint
    :param omega: decreasing parameter as defined in DUST
    :param beta_fea: Feasibility threshold in DUST (\beta_v)
    :param beta_opt: Optimality threshold in DUST (\beta_{\phi})
    :param theta: shrink parameter for rho
    :param max_iter: max number of iterations
    :param eig_add_on: add-on small constant to eigenvalue
    :param eta: initial eta (optional)
    :param lam: initial lam (optional)
    :param verbose: print out iteration information
    :return: (eta, current_opt_x, lam, current_fea_x, rho, ratio_complementary, ratio_opt, ratio_fea, iter_num, H_rho)
    """
    equatn = equatn.flatten()
    num_var = raw_H_0.shape[0]
    num_constraints = A.shape[0]

    # Set up dual problem
    # Constant vectors for the dual box constraints
    l = -1 * np.ones((num_constraints, 1))
    l[np.logical_not(equatn), 0] = 0
    c = np.ones((num_constraints, 1))

    # Prepare input for coordinate descent
    if eta is None:
        eta = np.zeros((num_constraints, 1))
    if lam is None:
        lam = np.zeros((num_constraints, 1))

    H_rho, L_rho, A_L_rho, dual_opt_obj_grad, v_eta, diag_rho = update_rho(g, A, b, raw_H_0, raw_H_f, rho, eta,
                                                                           eig_add_on)
    H_0, L_0, A_L_0, dual_fea_obj_grad, v_lam, diag_0 = update_rho(g, A, b, raw_H_0, raw_H_f, 0, lam, eig_add_on)

    cord_inds = np.arange(num_constraints)
    iter_num = 0
    current_fea_x = np.zeros((num_var, 1))
    current_opt_x = np.zeros((num_var, 1))
    j_0_w = get_primal_obj(H_rho, rho, g, A, b, current_opt_x, equatn) + omega
    ratio_complementary, ratio_opt, ratio_fea = (0, 0, 0)

    if verbose:
        print '''{0:3s} | {1:12s} | {2:12s} | {3:12s} | {4:12s} | {5:12s} | {6:12s} | {7:12s} | {8:12s} '''.format(
            'Itr', 'Primal_Opt', 'Dual_Opt', 'Primal_Fea', 'Dual_Fea', 'Ratio_Fea', 'Ratio_Opt', 'Ratio_C', 'Rho')

    # Here we solve a subproblem
    while iter_num < max_iter:
        np.random.shuffle(cord_inds)

        for i in cord_inds:
            # Update of optimality problem
            directional_opt_grad = -A_L_rho[[i], :].dot(v_eta)[0, 0] + dual_opt_obj_grad[i, 0]
            eta_i = 0
            if diag_rho[i, 0] == 0:
                if directional_opt_grad < 0:
                    eta_i = l[i, 0]
                elif directional_opt_grad > 0:
                    eta_i = c[i, 0]
                else:
                    eta_i = l[i, 0]
            else:
                tmp = directional_opt_grad / diag_rho[i, 0] + eta[i, 0]
                eta_i = min(max(tmp, l[i, 0]), c[i, 0])

            delta_eta = eta_i - eta[i, 0]
            eta[i, 0] = eta_i
            v_eta += delta_eta * A_L_rho[[i], :].T

            # Update of feasibility problem
            directional_fea_grad = -A_L_0[[i], :].dot(v_lam)[0, 0] + dual_fea_obj_grad[i, 0]
            lam_i = 0
            if diag_0[i, 0] == 0:
                if directional_fea_grad < 0:
                    lam_i = l[i, 0]
                elif directional_opt_grad > 0:
                    lam_i = c[i, 0]
                else:
                    lam_i = l[i, 0]
            else:
                tmp = directional_fea_grad / diag_0[i, 0] + lam[i, 0]
                lam_i = min(max(tmp, l[i, 0]), c[i, 0])

            delta_lam = lam_i - lam[i, 0]
            lam[i, 0] = lam_i
            v_lam += delta_lam * A_L_0[[i], :].T

        current_fea_x = get_primal_x(A, g, 0, lam, L_0)
        current_opt_x = get_primal_x(A, g, rho, eta, L_rho)
        # Get dual and primal objective function values
        obj_primal_opt = get_primal_obj(H_rho, rho, g, A, b, current_opt_x, equatn)
        obj_dual_opt = get_dual_obj(A, b, rho, g, L_rho, eta)
        obj_primal_fea = get_primal_obj(H_0, 0, g, A, b, current_opt_x, equatn)
        obj_dual_fea = get_dual_obj(A, b, 0, None, L_0, lam)

        # DUST update
        chi_d_u, l_d_0 = get_chi_du_l_d(A, b, current_opt_x, eta, equatn, num_constraints)

        # comment: what if j_0_w > 0 due to omega > 0
        ratio_fea = (j_0_w - l_d_0) / (j_0_w - max(0, obj_dual_fea))
        ratio_opt = (j_0_w - obj_primal_opt) / (j_0_w - obj_dual_opt)
        if j_0_w < chi_d_u:
            ratio_complementary = 0
        else:
            ratio_complementary = np.sqrt((j_0_w - chi_d_u) / j_0_w)

        if verbose:
            print '''{0:3d} | {1:+.5e} | {2:+.5e} | {3:+.5e} | {4:+.5e} | {5:+.5e} | {6:+.5e} | {7:+.5e} | {8:+.5e}''' \
                .format(iter_num, obj_primal_opt, obj_dual_opt, obj_primal_fea, obj_dual_fea, ratio_fea,
                        ratio_opt, ratio_complementary, rho)

        iter_num += 1
        if ratio_complementary >= beta_fea and ratio_opt >= beta_opt and ratio_fea >= beta_fea:
            break
        elif ratio_complementary >= beta_fea and ratio_opt >= beta_opt:
            rho *= theta
            H_rho, L_rho, A_L_rho, dual_opt_obj_grad, v_eta, diag_rho \
                = update_rho(g, A, b, raw_H_0, raw_H_f, rho, eta, eig_add_on)
            H_0, L_0, A_L_0, dual_fea_obj_grad, v_lam, diag_0 \
                = update_rho(g, A, b, raw_H_0, raw_H_f, 0, lam, eig_add_on)
        else:
            continue

    return eta, current_opt_x, lam, current_fea_x, rho, ratio_complementary, ratio_opt, ratio_fea, iter_num, H_rho


def setup_test():
    """
    Toy example to test coordinate descent
    :return:
    """
    num_var = 20
    num_constraints = 5
    A = np.random.normal(0, 1, (num_constraints, num_var))
    H_0 = np.random.normal(0, 1, (num_var, num_var))
    H_0 = H_0.T.dot(H_0)
    H_0 = positive_definify(H_0, 0.01)
    H_f = np.random.normal(0, 1, (num_var, num_var))
    H_f = H_f.T.dot(H_f)
    H_f = positive_definify(H_f, 0.01)
    b = np.random.normal(0, 1, (num_constraints, 1))
    g = np.random.normal(0, 1, (num_var, 1))
    equatn = np.array([True, True, False, True, False])
    omega = 0.01
    beta_fea = 0.2
    beta_opt = 0.8
    max_iter = 100
    rho = 1
    theta = 0.96

    print "H_0 cond : {0:+.3e}, H_rho cond : {1:+.3e}".format(np.linalg.cond(H_0), np.linalg.cond(H_0 + rho * H_f))

    eta, current_opt_x, lam, current_fea_x, rho = \
        cord_descent(H_0, H_f, rho, g, A, b, equatn, omega, beta_fea, beta_opt, theta, max_iter)


if __name__ == '__main__':
    setup_test()
