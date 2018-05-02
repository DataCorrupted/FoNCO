class DustParam:
    """
    Store all the dust linear solver parameters
    """

    def __init__(self, init_rho=1, init_omega=1e-2, max_iter=200, max_sub_iter=2000, beta_opt=0.7, beta_fea=0.1,
                 theta=0.9, line_theta= 1e-4, omega_shrink=0.7, eps_opt=1e-4, eps_violation=1e-5,
                 sub_verbose=False, rescale=True, SIGMA = 0.3, DELTA = 0.75, 
                 MIN_delta = 1e-5, MAX_delta = 64):
        self.init_rho = init_rho
        self.init_omega = init_omega
        self.max_iter = max_iter
        self.max_sub_iter = max_sub_iter
        self.beta_opt = beta_opt
        self.beta_fea = beta_fea
        self.theta = theta
        self.line_theta = line_theta
        self.omega_shrink = omega_shrink
        self.eps_opt = eps_opt
        self.eps_violation = eps_violation
        self.sub_verbose = sub_verbose
        self.rescale = rescale
        self.SIGMA = SIGMA
        self.DELTA = DELTA
        self.MIN_delta = MIN_delta
        self.MAX_delta = MAX_delta