class DustParam:
    """
    Store all the dust linear solver parameters
    """

    def __init__(self, **kwarg):

        # Setup default ones.
        # I didn't use default value so that I can see clearer.

        self.init_delta = 1;
        self.init_rho = 1;
        self.init_omega = 1e-2;
        self.max_iter = 512;
        self.max_sub_iter = 100;
        self.beta_opt = 0.75;
        self.beta_fea = 0.3;

        #TODO Try smaller theta
        self.theta = 0.9;
        self.line_theta =  1e-4;
        self.omega_shrink = 0.7;
        self.eps_opt = 1e-4;
        self.eps_violation = 1e-3;
        self.sub_verbose = False;
        self.rescale = True;
        self.SIGMA = 0.3;               # Trust region update.
        self.DELTA = 0.75;              # Trust region update.
        self.MIN_delta = 1e-2;          # Min trust region.    
        self.MAX_delta = 64;            # Max trust region.    

        # Take in user's request.
        if 'init_rho' in kwarg:         self.init_rho = kwarg['init_rho'];
        if 'init_omega' in kwarg:       self.init_omega = kwarg['init_omega'];
        if 'init_delta' in kwarg:       self.init_delta = kwarg['init_delta']

        if 'max_iter' in kwarg:         self.max_iter = kwarg['max_iter'];
        if 'max_sub_iter' in kwarg:     self.max_sub_iter = kwarg['max_sub_iter'];
        if 'beta_opt' in kwarg:         self.beta_opt = kwarg['beta_opt'];
        if 'beta_fea' in kwarg:         self.beta_fea = kwarg['beta_fea'];
        if 'theta' in kwarg:            self.theta = kwarg['theta'];
        if 'line_theta' in kwarg:       self.line_theta = kwarg['line_theta'];
        if 'omega_shrink' in kwarg:     self.omega_shrink = kwarg['omega_shrink'];
        if 'eps_opt' in kwarg:          self.eps_opt = kwarg['eps_opt'];
        if 'eps_violation' in kwarg:    self.eps_violation = kwarg['eps_violation'];
        if 'sub_verbose' in kwarg:      self.sub_verbose = kwarg['sub_verbose'];
        if 'rescale' in kwarg:          self.rescale = kwarg['rescale'];
        if 'SIGMA' in kwarg:            self.SIGMA = kwarg['SIGMA'];
        if 'DELTA' in kwarg:            self.DELTA = kwarg['DELTA'];
        if 'MIN_delta' in kwarg:        self.MIN_delta = kwarg['MIN_delta'];
        if 'MAX_delta' in kwarg:        self.MAX_delta = kwarg['MAX_delta'];

    def dump2Dict(self):
        return {
            "init_rho": self.init_rho,
            "init_omega": self.init_omega,
            "init_delta": self.init_delta,
            
            "max_iter": self.max_iter,
            "max_sub_iter": self.max_sub_iter,
            "beta_opt": self.beta_opt,
            "beta_fea": self.beta_fea,

            "theta": self.theta,
            "line_theta": self.line_theta,
            "omega_shrink": self.omega_shrink,
            "eps_opt": self.eps_opt,
            "eps_violation": self.eps_violation,
            "sub_verbose": self.sub_verbose,
            "rescale": self.rescale,
            "SIGMA": self.SIGMA,
            "DELTA": self.DELTA,
            "MIN_delta": self.MIN_delta,
            "MAX_delta": self.MAX_delta
        }