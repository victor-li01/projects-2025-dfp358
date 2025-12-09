from types import SimpleNamespace

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

class WorkerClass:

    def __init__(self,par=None):
        # If a parameter dict is provided, populate `self.par` first
        # so that `setup_worker` can respect values (like `sigma`) provided
        if par is None:
            self.par = None
        else:
            self.par = SimpleNamespace()
            for k, v in par.items():
                setattr(self.par, k, v)

        # a. setup (will only fill defaults for missing attributes)
        self.setup_worker()

        # b. update parameters (keep compatibility with callers)
        if not par is None:
            for k, v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):

        # preserve any pre-set `self.par` (e.g. passed into __init__)
        if not hasattr(self, 'par') or self.par is None:
            par = self.par = SimpleNamespace()
        else:
            par = self.par

        # ensure solution namespace exists
        if not hasattr(self, 'sol') or self.sol is None:
            sol = self.sol = SimpleNamespace()
        else:
            sol = self.sol

        # a. preferences (only set defaults if not already provided)
        if not hasattr(par, 'nu'):
            par.nu = 0.015 # weight on labor disutility
        if not hasattr(par, 'epsilon'):
            par.epsilon = 1.0 # curvature of labor disutility
        if not hasattr(par, 'sigma'):
            par.sigma = 1.0
        
        # b. productivity and wages
        par.w = 1.0 # wage rate
        par.ps = np.linspace(0.5,3.0,100) # productivities
        par.ell_max = 16.0 # max labor supply
        
        # c. taxes
        par.tau = 0.50 # proportional tax rate
        par.zeta = 0.10 # lump-sum tax
        par.kappa = np.nan # income threshold for top tax
        par.omega = 0.20 # top rate rate
          
    def utility(self,c,ell):

        par = self.par
        
        if par.sigma == 1.0:
            u_c = np.log(c)
        else:
            u_c = (c**(1-par.sigma)-1)/(1-par.sigma)
      
        u_ell = par.nu * (ell**(1+par.epsilon))/(1+par.epsilon)
        u = u_c - u_ell
        
        return u
    
    def tax(self,pre_tax_income):

        par = self.par

        tax = par.tau * pre_tax_income + par.zeta
        
        if not np.isnan(par.kappa):
            top_income = np.fmax(pre_tax_income - par.kappa,0.0)
            tax += par.omega * top_income

        return tax
    
    def income(self,p,ell):

        par = self.par
        
        income = par.w * p * ell

        return income

    def post_tax_income(self,p,ell):

        pre_tax_income = self.income(p,ell)
        tax = self.tax(pre_tax_income)

        return pre_tax_income - tax
    
    def max_post_tax_income(self,p):

        par = self.par
        return self.post_tax_income(p,par.ell_max)

    def value_of_choice(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        U = self.utility(c,ell)

        return U
    
    def get_min_ell(self,p):
    
        par = self.par

        min_ell = par.zeta/(par.w*p*(1-par.tau))
        
        if not np.isnan(par.kappa): assert par.zeta <= 0.0, "Only lump-sum transfers allowed with tax kink"
        
        assert min_ell < par.ell_max, "Lump-sum tax too high, no feasible labor supply"

        return np.fmax(min_ell,0.0) + 1e-8
    
    def optimal_choice(self,p): # Solving via numerical optimization

        par = self.par
        opt = SimpleNamespace()

        # a. objective function
        obj = lambda ell: -self.value_of_choice(p,ell)

        # b. bounds and minimization
        bounds = (self.get_min_ell(p),par.ell_max)
        res = minimize_scalar(obj,bounds=bounds,method='bounded')

        # c. results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p,opt.ell)

        return opt
    
    def FOC(self,p,ell, omega=0.0):

        par = self.par

        # a: consumption
        c = self.post_tax_income(p,ell)
        
        # b: marginal utilities
        if par.sigma == 1.0:
            dU_dc = 1.0 / c
        else:
            dU_dc = c ** (-par.sigma)
            
        dc_dell = par.w * p * (1 - par.tau-omega)
        dU_dell = - par.nu * (ell**par.epsilon)
        
        # c: FOC
        FOC = dU_dc * dc_dell + dU_dell
        
        return FOC
    
    def optimal_choice_FOC(self, p):

        par = self.par
        opt = SimpleNamespace()
        
        # a: no kink
        if np.isnan(par.kappa):
            
            # i: objective function
            obj = lambda ell: self.FOC(p, ell)
            
            # ii: bounds and root finding
            min_ell = self.get_min_ell(p)
            max_ell = par.ell_max
            
            if obj(min_ell) * obj(max_ell) < 0:
                
                bounds = (min_ell, max_ell)
                res = root_scalar(obj, bracket=bounds, method='bisect')
                opt.ell = res.root
                
            else:
                opt.ell = par.ell_max
            
            # iii: results
            opt.U = self.value_of_choice(p, opt.ell)
            opt.c = self.post_tax_income(p, opt.ell)
            opt.section = None
            
            return opt
        
        else:
            
            ell_kink = par.kappa / (par.w * p)
            
            # i: objective functions
            FOC_below = lambda ell: self.FOC(p, ell, omega=0.0)
            FOC_above = lambda ell: self.FOC(p, ell, omega=par.omega)
            
            # ii: bounds and root finding
            min_ell = self.get_min_ell(p)
            
            # Ab hier ohne Lösung, nehme von ChatGPT
            
            # --- Step 1: below kink ---
            U_b = -np.inf
            ell_b = None
            if min_ell < ell_kink and FOC_below(min_ell) * FOC_below(ell_kink) < 0:
                res = root_scalar(FOC_below, bracket=(min_ell, ell_kink), method='bisect')
                ell_b = res.root
                U_b = self.value_of_choice(p, ell_b)

            # --- Step 2: kink point ---
            ell_k = ell_kink
            U_k = self.value_of_choice(p, ell_k)

            # --- Step 3: above kink ---
            U_a = -np.inf
            ell_a = None
            if par.ell_max > ell_kink and FOC_above(ell_kink) * FOC_above(par.ell_max) < 0:
                res = root_scalar(FOC_above, bracket=(ell_kink, par.ell_max), method='bisect')
                ell_a = res.root
                U_a = self.value_of_choice(p, ell_a)

            # --- Step 4: pick best ---
            U_list = [U_b, U_k, U_a]
            ell_list = [ell_b, ell_k, ell_a]
            sections = ["below", "kink", "above"]

            best_idx = np.argmax(U_list)

            opt.ell = ell_list[best_idx]
            opt.U   = U_list[best_idx]
            opt.c   = self.post_tax_income(p, opt.ell)
            opt.section = sections[best_idx]

            return opt

   


    


        
        