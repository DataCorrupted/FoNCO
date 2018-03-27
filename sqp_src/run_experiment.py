#!/bin/env python
import os
from collections import OrderedDict

import numpy as np
import logging
import pickle
import time

from cuter_util import Cuter
from non_linear_solver import non_linear_solve, DustParam


def get_logger(log_dir, log_file, logLevel=logging.DEBUG):
    """
    Create an logging instance with log stored under `log_dir`/`log_file`
    :param log_dir: log directory created it if not existed
    :param log_file: log file name
    :param logLevel: logging level default to be debug
    :return: logging instance
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logLevel)
    logger = logging.getLogger(log_file)
    log_path = os.path.join(log_dir, log_file)
    fh = logging.FileHandler(log_path, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s %(name)-4s %(levelname)-2s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def print_problem_statement(setup_args_dict, logger):
    """
    Print the cuter problem information
    :param setup_args_dict: set up parameter dictionary
    :param logger: stream to logger instead of print to screen if not None
    :return: None
    """
    equatn = setup_args_dict['equatn']
    num_eq = np.sum(equatn)
    bu_flag = setup_args_dict['bu_flag']
    bl_flag = setup_args_dict['bl_flag']
    nlow = np.sum(bl_flag)
    nup = np.sum(bu_flag)
    obj_scale = setup_args_dict['obj_scale']

    str_format = '| Var #:{0:6d} | Con #:{1:6d} | Eq #:{2:6d} | Low_bound #:{3:6d} | Up_bound #:{4:6d} | Obj_scale: {5:+.5e}' \
        .format(setup_args_dict['n'][0], setup_args_dict['m'][0], num_eq, nlow, nup, obj_scale)

    if logger is None:
        print str_format
    else:
        logger.info(str_format)


def print_param_dict(param_dict, logger=None):
    """
    Print out param dictionary
    :param param_dict: parameter dictionary
    :param logger: stream to logger instead of print to screen if not None
    :return:
    """
    key_len = max([len(k) for k in param_dict.keys()]) + 5
    str_format = "{{0:{0}s}} : {{1}}".format(key_len)
    for k, v in param_dict.items():
        if logger is None:
            print str_format.format(k, v)
        else:
            logger.info(str_format.format(k, v))


def save_output(result_dir, dust_output):
    """
    Serialize dust_output to result_dir with file name as defined by dust_output['problem_name']
    :param result_dir: output directory
    :param dust_output: dust output
    :return: None
    """

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    outfile = os.path.join(result_dir, '{0}.pkl'.format(dust_output['problem_name']))

    with open(outfile, 'w') as f:
        pickle.dump(dust_output, f)


def nlp_test(sif_dir_root, problem_name, dust_param, log_dir, result_dir):
    """
    Test one cuter problem stored under `sif_dir_root`/`problem_name`
    :param sif_dir_root: sif directory
    :param problem_name: problem name, e.g. HS101
    :param dust_param: dust param instance
    :param log_dir: output log directory
    :param result_dir: result directory to store the output
    :return: dust output
    """
    logger = get_logger(log_dir, '{0}.log'.format(problem_name))
    logger.info('+' * 200)
    with Cuter(os.path.join(sif_dir_root, problem_name)) as cuter:
        try:
            logger.info("Problem name: {0}".format(problem_name))
            print_problem_statement(cuter.setup_args_dict, logger)
            logger.info('-' * 200)
            init_param_dict = {'Omega': dust_param.init_omega, 'Rho': dust_param.init_rho,
                               'Beta_opt': dust_param.beta_opt,
                               'Beta_feasibility': dust_param.beta_fea, 'Hessian add-on': dust_param.add_on_hess,
                               'Line search theta': dust_param.line_theta, 'Theta': dust_param.theta,
                               'Omega shrink': dust_param.omega_shrink, 'Max sub iteration': dust_param.max_sub_iter,
                               'Eps opt': dust_param.eps_opt}
            print_param_dict(init_param_dict, logger)
            start_time = time.time()
            dust_output = non_linear_solve(cuter, dust_param, logger)
            execution_time = time.time() - start_time
            dust_output['problem_name'] = problem_name
            dust_output['execution_time'] = execution_time
            output_print_dict = OrderedDict(
                [('Summary for problem', dust_output['problem_name']),
                 ('Status', dust_output['status']), ('Iteration Number', dust_output['iter_num']),
                 ('Final objective', dust_output['obj_f']), ('KKT error', dust_output['kkt_error']),
                 ('Constraint violation', dust_output['constraint_violation']), ('Execute Time (s)', execution_time),
                 ('Number Fn', dust_output['fn_eval_cnt'])])
            print_param_dict(output_print_dict, logger)
            save_output(result_dir, dust_output)
        except Exception as e:
            logger.error(e)
            logger.error('End of problem: {0}'.format(problem_name))

    logger.info('+' * 200)


def all_tests(sif_dir_root, log_dir, result_dir):
    """
    Run all tests
    :param sif_dir_root: directory of all sif files
    :param log_dir: output log directory
    :param result_dir: result directory to store the output
    :return:
    """
    skip_list = ['HS93', 'HS99EXP', 'HS114', 'HS68', 'HS116', 'HS83', 'HS13', 'HS84', 'HS85', 'HS87', 'HS106']
    problem_list = os.listdir(sif_dir_root)
    for problem_name in problem_list:
        if problem_name.startswith("HS") and problem_name not in skip_list:
            if problem_name == 'HS19':
                dust_param = DustParam(max_sub_iter=1e4)
                nlp_test(sif_dir_root, problem_name, dust_param, log_dir, result_dir)
            elif problem_name == 'HS75':
                # Problem HS75's Hessian's scale is on 1e-4, if add_on is on 1e-4, this changes the Hessian strucutre
                # dramatically, tune down the add_on to be 1e-8
                dust_param = DustParam(add_on_hess=1e-8)
                nlp_test(sif_dir_root, problem_name, dust_param, log_dir, result_dir)
            else:
                dust_param = DustParam()
                nlp_test(sif_dir_root, problem_name, dust_param, log_dir, result_dir)


if __name__ == '__main__':
    sif_dir_root = '/home/peter/Desktop/slp/sif'
    log_dir = './logs/logs_0'
    result_dir = './results/results_0'
    # all_tests(sif_dir_root, log_dir, result_dir)
    problem_name = 'HS89'
    dust_param = DustParam()
    all_tests(sif_dir_root, log_dir, result_dir)
