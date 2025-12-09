from Worker import WorkerClass
import numpy as np
from scipy.optimize import minimize

class GovernmentClass(WorkerClass):

    def __init__(self,par=None):

        # a. defaul setup
        self.setup_worker()
        self.setup_government()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. random number generator
        self.rng = np.random.default_rng(12345)
    
    # ⭐ ADD THIS ⭐ -> Von Chatty gucken wir mal ob wichtig später
    # def set_policy(self, tau, zeta): 
        # self.par.tau = tau
        # self.par.zeta = zeta

    def setup_government(self):

        par = self.par

        # a. workers
        par.N = 100  # number of workers
        par.sigma_p = 0.3  # std dev of productivity

        # b. pulic good
        par.chi = 50.0 # weight on public good in SWF
        par.eta = 0.1 # curvature of public good in SWF

    def draw_productivities(self):

        par = self.par
        
        par.ps = np.exp(self.rng.normal(-0.5*par.sigma_p**2, par.sigma_p, par.N))

    def solve_workers(self):

        par = self.par
        sol = self.sol
        
        sol.ell_opt = np.zeros(par.N)
        sol.U_opt = np.zeros(par.N)
        sol.c_opt = np.zeros(par.N)


        for i in range(par.N):
            
            p = par.ps[i]
            sol_i = self.optimal_choice_FOC(p)
            
            sol.ell_opt[i] = sol_i.ell
            sol.U_opt[i] = sol_i.U
            sol.c_opt[i] = sol_i.c


    def tax_revenue(self):

        par = self.par
        sol = self.sol

        # labor income: y_i = w * p_i * ell_i
        income = par.w * par.ps * sol.ell_opt

        # base tax revenue
        base = par.N * par.zeta + par.tau * np.sum(income)

        # top tax bracket revenue
        if not np.isnan(par.kappa):
            top_income = np.fmax(income - par.kappa, 0.0)
            top_tax = par.omega * np.sum(top_income)
        else:
            top_tax = 0.0

        tax_revenue = base + top_tax   # ← define variable

        return tax_revenue              

    
    
    def SWF(self):

        par = self.par
        sol = self.sol
        
        if np.any(sol.c_opt <= 0):
            return -1e12
        
        # 1: Compute tax revenue
        G =  self.tax_revenue()
        
        # Feasibility check: government budget cannot be negative
        if G < 0:
            return -1e12
        else:
            # 2. Compute public good utility term
            public_good_term = par.chi * (G**par.eta)

            # 3. Sum utilities of all workers
            util_sum = np.sum(sol.U_opt)

            # 4. Social welfare
            SWF = public_good_term + util_sum

        return SWF
    
    
    def optimal_taxes(self,tau,zeta):

        par = self.par

        # a. objective function
        def obj(x):
            
            tau  = x[0]
            zeta = x[1]

            par.tau = tau
            par.zeta = zeta

            # solve worker segment
            self.solve_workers()

            # compute welfare
            SWF = self.SWF()

            # minimization: return negative SWF
            return -SWF
        
        
        # --- b. bounds for (tau, zeta) ---
        bounds = [
        (0.0, 0.99),     # proportional tax rate
        (-1, 1)      # lump-sum (can be negative)
        ]

        # --- c. initial guess ---
        x0 = np.array([par.tau, par.zeta])

        # --- d. optimization ---
        res = minimize(
        obj,
        x0,
        bounds=bounds,
        method='L-BFGS-B'
        )

        # --- e. store optimal parameters ---
        par.tau  = res.x[0]
        par.zeta = res.x[1]
        SWF_opt  = -res.fun

        return par.tau, par.zeta, SWF_opt 
        
