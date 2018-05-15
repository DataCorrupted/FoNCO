#!/bin/env python

import os
from ctypes import *
import numpy as np

PtD = POINTER(c_double)
PtI = POINTER(c_int)
PtB = POINTER(c_bool)


class Cuter(object):
    __PMAX = 1e3
    __UP = 1e20
    __LOW = -1e20

    def __init__(self, problem_path):
        """
        problem path includes py_cute.so and OUTSDIF.d
        :param problem_path: cuter problem path
        """
        self.problem_path = problem_path
        self.__handler = None
        self.setup_args_dict = None

    def __enter__(self):
        self.__setup_cuter()
        return self

    def __exit__(self, *args):
        """
        Close handler
        :return: close the handler
        """
        self.__handler.CUTEST_cterminate_1(self.setup_args_dict['status'])
        self.__handler._close(self.setup_args_dict['funit'])

    def __setup_cuter(self):
        """
        Set up cuter problem
        :return:
        """
        self.__handler = CDLL(os.path.join(self.problem_path, 'py_cute.so'))

        status = PtI(c_int(0))
        funit = PtI(c_int(42))
        fname = c_char_p(os.path.join(self.problem_path, 'OUTSDIF.d'))
        iout = PtI(c_int(6))
        io_buffer = PtI(c_int(11))
        e_order = PtI(c_int(1))
        l_order = PtI(c_int(1))
        v_order = PtI(c_int(2))
        n = PtI(c_int(0))
        m = PtI(c_int(0))

        # Open the file and get the dimension of the problem
        self.__handler._open(fname, funit)
        self.__handler.CUTEST_cdimen_1(status, funit, n, m)

        x = np.zeros((n[0], 1))
        bl = np.zeros((n[0], 1))
        bu = np.zeros((n[0], 1))
        v = np.zeros((m[0], 1))
        cl = np.zeros((m[0], 1))
        cu = np.zeros((m[0], 1))
        equatn = np.array([0] * m[0], dtype=np.bool)
        linear = np.array([0] * m[0], dtype=np.bool)

        # Setup the problem
        self.__handler.CUTEST_csetup_1(status, funit, iout, io_buffer, n, m,
                                       x.ctypes.data_as(PtD),
                                       bl.ctypes.data_as(PtD),
                                       bu.ctypes.data_as(PtD),
                                       v.ctypes.data_as(PtD),
                                       cl.ctypes.data_as(PtD),
                                       cu.ctypes.data_as(PtD),
                                       equatn.ctypes.data_as(POINTER(c_bool)),
                                       linear.ctypes.data_as(POINTER(c_bool)),
                                       e_order, l_order, v_order)

        iequatn = np.logical_not(equatn)
        inequality_upper = np.logical_and((cu != self.__UP).flatten(), iequatn)
        inequality_lower = np.logical_and((cl != self.__LOW).flatten(), iequatn)
        is_lower_bound_only_constr = np.logical_and(inequality_lower, np.logical_not(inequality_upper))
        is_double_bound_constr = np.logical_and(inequality_lower, inequality_upper)
        bl_flag = (bl != self.__LOW).flatten()
        bu_flag = (bu != self.__UP).flatten()
        num_added_ineq_constr = np.sum(is_double_bound_constr) + np.sum(bl_flag) + np.sum(bu_flag)
        added_iequatn = np.array([False] * num_added_ineq_constr, dtype=bool)
        adjusted_equatn = np.hstack([equatn, added_iequatn])

        self.setup_args_dict = {'status': status,
                                'funit': funit,
                                'iout': iout,
                                'n': n,
                                'm': m,
                                'x': x,
                                'bl': bl,
                                'bu': bu,
                                'v': v,
                                'cl': cl,
                                'cu': cu,
                                'equatn': equatn,
                                'linear': linear,
                                'e_order': e_order,
                                'l_order': l_order,
                                'v_order': v_order,
                                'bl_flag': bl_flag,
                                'bu_flag': bu_flag,
                                'adjusted_equatn': adjusted_equatn,
                                'iequatn': iequatn,
                                'inequality_lower': inequality_lower,
                                'inequality_upper': inequality_upper,
                                'is_lower_bound_only_constr': is_lower_bound_only_constr,
                                'is_double_bound_constr': is_double_bound_constr}

        self.__get_rescale_factors()

    def __get_rescale_factors(self):
        """
        Calculate rescale factor for the problem
        :return: None, update setup_args_dict to include rescale factors in-place
        """
        setup_args_dict = self.setup_args_dict
        x_0 = self.setup_args_dict['x']

        f, g = self.get_f_g(x_0, grad_flag=True)
        A, b = self.get_constr_f_g(x_0, grad_flag=True)

        # A is in extended canonical form, get the first `m` constraints which are original c_i(x)
        num_constr = setup_args_dict['m'][0]
        jacob = A[:num_constr]

        obj_scale = self.__PMAX / max(np.linalg.norm(g, ord=np.inf), self.__PMAX)
        jacob_norm = np.linalg.norm(jacob, ord=np.inf, axis=1)
        jacob_norm[jacob_norm < self.__PMAX] = self.__PMAX
        constr_scale = self.__PMAX / jacob_norm
        constr_scale = constr_scale.reshape((-1, 1))

        setup_args_dict['obj_scale'] = obj_scale
        setup_args_dict['constr_scale'] = constr_scale

    def get_f_g(self, x, grad_flag=False, rescale=False):
        """
        Evaluate the function value and possibly gradient of objective at x
        :param x: current x
        :param grad_flag: indicator of evaluate gradient of objective
        :param rescale: boolean indicator whether to apply rescale to the problem
        :return: f(x) and f'(x) if grad_flag True
        """
        setup_args_dict = self.setup_args_dict
        if grad_flag:
            grad = POINTER(c_bool)(c_bool(1))
        else:
            grad = POINTER(c_bool)(c_bool(0))
        f = PtD(c_double(0))
        status = self.setup_args_dict['status']
        nvar = self.setup_args_dict['n']
        g = np.zeros((nvar[0], 1), order='F')
        self.__handler.CUTEST_cofg_1(status, nvar, x.ctypes.data_as(PtD), f, g.ctypes.data_as(PtD), grad)
        f = f[0]

        if rescale:
            obj_scale = setup_args_dict['obj_scale']
            f *= obj_scale
            g *= obj_scale

        if grad_flag:
            return f, g
        else:
            return f, None

    def get_constr_f_g(self, x, grad_flag=False, rescale=False):
        """
        Evaluate the function value and possibly gradient (Jacobian) of constraints at x
        :param x: current x
        :param grad_flag: indicator of evaluate gradient of objective
        :param rescale: boolean indicator whether to apply rescale to the problem
        :return: constr_f(x) and possibly J(x) if grad_flag True
        """
        setup_args_dict = self.setup_args_dict
        if grad_flag:
            grad = POINTER(c_bool)(c_bool(1))
        else:
            grad = POINTER(c_bool)(c_bool(0))

        nvar = setup_args_dict['n'][0]
        c = np.zeros((setup_args_dict['m'][0], 1), order='F')
        jtrans = POINTER(c_bool)(c_bool(0))
        lcjac1 = POINTER(c_int)(c_int(setup_args_dict['m'][0]))
        lcjac2 = POINTER(c_int)(c_int(setup_args_dict['n'][0]))
        # Big bug Fortran order
        cjac = np.zeros((setup_args_dict['m'][0], setup_args_dict['n'][0]), order='F')
        self.__handler.CUTEST_ccfg_1(setup_args_dict['status'],
                                     setup_args_dict['n'],
                                     setup_args_dict['m'],
                                     x.ctypes.data_as(PtD),
                                     c.ctypes.data_as(PtD),
                                     jtrans,
                                     lcjac1,
                                     lcjac2,
                                     cjac.ctypes.data_as(PtD),
                                     grad)

        if rescale:
            constr_scale = setup_args_dict['constr_scale']
            c *= constr_scale
            cjac *= constr_scale

        inequality_upper = setup_args_dict['inequality_upper']
        equatn = setup_args_dict['equatn']
        # Make all constraints to be <=
        cl = setup_args_dict['cl']
        cu = setup_args_dict['cu']
        # c_i(x) - cu_i <= 0 for inequality constraint and cu_i < UP
        c[inequality_upper] -= cu[inequality_upper]
        c[equatn] -= cl[equatn]

        # For the general constraints with only lower bound, convert it into format cl_i - c_i(x) <= 0
        is_lower_bound_only_constr = setup_args_dict['is_lower_bound_only_constr']
        c[is_lower_bound_only_constr] = cl[is_lower_bound_only_constr] - c[is_lower_bound_only_constr]
        cjac[is_lower_bound_only_constr] *= -1

        # Add bound constraints
        bl_flag = setup_args_dict['bl_flag']
        bu_flag = setup_args_dict['bu_flag']
        bu = setup_args_dict['bu']
        bl = setup_args_dict['bl']
        bound_constr_f_upper = x[bu_flag] - bu[bu_flag]
        bound_constr_f_lower = bl[bl_flag] - x[bl_flag]

        # Add constraints cl_i - c_i(x) <= 0 for inequality constraint and cl_i >= LOW when there are both lower
        # and upper bounds for this constraint
        # cl <= c(x) <= cu for those with both lower and upper bound, we need to add one more row to Jacobian and
        # constraint function evaluation
        is_double_bound_constr = setup_args_dict['is_double_bound_constr']
        c_double_bound = cl[is_double_bound_constr] - c[is_double_bound_constr]
        final_c = np.vstack([c, c_double_bound, bound_constr_f_upper, bound_constr_f_lower])
        cjac_double_bound = cjac[is_double_bound_constr] * -1
        cjac = np.vstack([cjac, cjac_double_bound])

        if grad_flag:
            # Add bound constraints to Jacobian (upper bound before lower bound)
            upper_bound_jac = np.zeros((np.sum(bu_flag), nvar))
            col_ind_upper = np.where(bu_flag)[0]
            row_ind_upper = np.arange(col_ind_upper.shape[0])
            upper_bound_jac[row_ind_upper, col_ind_upper] = 1

            lower_bound_jac = np.zeros((np.sum(bl_flag), nvar))
            col_ind_lower = np.where(bl_flag)[0]
            row_ind_lower = np.arange(col_ind_lower.shape[0])
            lower_bound_jac[row_ind_lower, col_ind_lower] = -1
            cjac = np.vstack([cjac, upper_bound_jac, lower_bound_jac])
            return final_c, cjac
        else:
            return final_c, None

    def get_hessian(self, x, iprob, rescale=False):
        """
        Calculate the Hessian of objective function at x or i-th constraint
        :param x: current x
        :param iprob: index, 0: objective, i > 0: ith constraint
        :param rescale: boolean indicator whether to apply rescale to the problem
        :return: H_f(x)
        """
        setup_args_dict = self.setup_args_dict
        status = setup_args_dict['status']
        nvar = setup_args_dict['n']
        hessian_f = np.zeros((nvar[0], nvar[0]), order='F')
        ind_prob = PtI(c_int(iprob))
        self.__handler.CUTEST_cidh_1(status,
                                     nvar,
                                     x.ctypes.data_as(PtD),
                                     ind_prob,
                                     nvar,
                                     hessian_f.ctypes.data_as(PtD)
                                     )

        if rescale:
            if iprob == 0:
                obj_scale = setup_args_dict['obj_scale']
                hessian_f *= obj_scale
            else:
                constr_scale = setup_args_dict['constr_scale']
                hessian_f *= constr_scale[iprob - 1]

        return hessian_f

    def get_hessian_lagrangian(self, x, multiplier_lagrangian, rescale=False):
        """
        Calculate the Hessian of Lagrangian at x with dual estimation `dual_var`
        :param x: current x
        :param multiplier_lagrangian: current dual estimation (dimension should be `setup_args_dict['m']`)
        :param rescale: boolean indicator whether to apply rescale to the problem
        :return: H_l(x)
        """
        setup_args_dict = self.setup_args_dict
        status = setup_args_dict['status']
        nvar = setup_args_dict['n']
        m = setup_args_dict['m']
        hessian_l = np.zeros((nvar[0], nvar[0]), order='F')

        if rescale:
            constr_scale = setup_args_dict['constr_scale']
            multiplier_lagrangian *= constr_scale
            obj_scale = setup_args_dict['obj_scale']
            multiplier_lagrangian /= obj_scale

        self.__handler.CUTEST_cdh_1(status,
                                    nvar,
                                    m,
                                    x.ctypes.data_as(PtD),
                                    multiplier_lagrangian.ctypes.data_as(PtD),
                                    nvar,
                                    hessian_l.ctypes.data_as(PtD)
                                    )

        if rescale:
            obj_scale = setup_args_dict['obj_scale']
            hessian_l *= obj_scale

        return hessian_l

    def dual_var_adapter(self, dual_var):
        """
        Convert `dual_var` into the Lagrangian multiplier used by cuter Hessian subroutine
        :param dual_var: dual variables in canonical form, all inequality constraints to be c(x) <= 0 and
                         all bound constraints converted to general constrains
        :return: Lagrangian multiplier used by cuter Hessian subroutine
        """
        setup_args_dict = self.setup_args_dict
        nconstr = setup_args_dict['m'][0]
        is_lower_bound_only_constr = setup_args_dict['is_lower_bound_only_constr']
        is_double_bound_constr = setup_args_dict['is_double_bound_constr']

        multiplier_lagrangian = dual_var[:nconstr].copy()
        # Convert back to multiplier for c_i(x) >= cl_i
        multiplier_lagrangian[is_lower_bound_only_constr] *= -1

        # For cl_i <= c_i(x) <= cu_i, we add a constraint -c_i(x) + cl_i <= 0 after the original general constraints
        # Let eta_i be multiplier for c_i(x) <= cu_i and d_i be multiplier for -c_i(x) + cl_i <= 0
        # In the cuter Hessian subroutine setting, the multiplier for Hessisan(c_i(x)) should be (eta_i - d_i)
        double_bound_constr_index = np.arange(np.sum(is_double_bound_constr)) + nconstr
        multiplier_lagrangian[is_double_bound_constr] -= dual_var[double_bound_constr_index]

        return multiplier_lagrangian
