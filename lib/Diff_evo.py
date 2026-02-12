import pandas as pd
import numpy as np

class EvolutionAlgorithmBase:
    """
    Base class for Evolution Algorithms.
    """
    def __init__(self, func, n_dim, size_pop, max_iter, prob_mut):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut
        
        # History containers
        self.generation_best_X = []
        self.generation_best_Y = []
        self.all_history_Y = []
        self.all_history_FitV = []

    def run(self):
        pass


class DE(EvolutionAlgorithmBase):
    """
    Differential Evolution (DE) Algorithm.
    
    This class implements the Differential Evolution algorithm for activation cutoff optimization.
    It uses a loop-based approach to ensure distinct candidate selection for mutation.
    This implementation maximizes the accuracy.
    
    Parameters:
    -----------
    func: callable
        The objective function to minimize. 
    n_dim: int
        The dimension of the search space (Layers*blocks*2 + 2) = 20.
    F: float
        The mutation factor (differential weight).
    size_pop: int
        The size of the population.
    max_iter: int
        The maximum number of iterations/generations.
    lb: array
        Lower bounds for the activation values.
    ub: array
        Upper bounds for the activation values.
    prob_mut: float (optional, default 0.7)
        The crossover rate (CR).
    """
    def __init__(self, func, F, lb, ub,
                 size_pop, n_dim, max_iter, prob_mut):
        # Note: 'prob_mut' corresponds to 'cr' (crossover rate) in sample code
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut)

        self.F = F
        self.lb = np.array(lb) * np.ones(self.n_dim)
        self.ub = np.array(ub) * np.ones(self.n_dim)
        
        # Initialize population
        self.crtbp()
        # Evaluate initial population
        self.Y = np.array([self.func(x) for x in self.X])

    def crtbp(self):
        """Create the initial population randomly within bounds"""
        self.X = self.lb + (np.random.rand(self.size_pop, self.n_dim) * (self.ub - self.lb))
        return self.X

    def mutation_op(self, x, F):
        """
        Mutation operation: x[0] + F * (x[1] - x[2])
        x is a list/array of 3 vectors [a, b, c]
        """
        return x[0] + F * (x[1] - x[2])

    def check_bounds(self, mutated):
        """Boundary check operation using clip"""
        return np.clip(mutated, self.lb, self.ub)

    def crossover_op(self, mutated, target, cr):
        """
        Crossover operation
        """
        # generate a uniform random value for every dimension
        p = np.random.rand(self.n_dim)
        # generate trial vector by binomial crossover
        # trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
        # Vectorized version for single individual:
        trial = np.where(p < cr, mutated, target)
        return trial

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        
        # Initial Best
        best_idx = np.argmax(self.Y)
        self.best_x = self.X[best_idx].copy()
        self.best_y = self.Y[best_idx]

        for i in range(self.max_iter):
            # Iterate over all candidate solutions
            for j in range(self.size_pop):
                # Choose three candidates a, b, c that are not the current one
                # to ensure distinct indices for mutation
                candidates = [idx for idx in range(self.size_pop) if idx != j]
                a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
                
                a = self.X[a_idx]
                b = self.X[b_idx]
                c = self.X[c_idx]
                
                # Perform mutation
                mutated = self.mutation_op([a, b, c], self.F)
                
                # Check bounds
                mutated = self.check_bounds(mutated)
                
                # Perform crossover
                trial = self.crossover_op(mutated, self.X[j], self.prob_mut)
                
                # Compute objective function value for trial vector
                # (Assuming func takes a single vector)
                if hasattr(self.func, 'batch_mode') and self.func.batch_mode:
                     # Handle batch if necessary, but sample assumes single
                     obj_trial = self.func(trial.reshape(1, -1))[0]
                else:
                     obj_trial = self.func(trial)
                
                obj_target = self.Y[j]
                
                # Perform selection
                if obj_trial > obj_target:
                    # Replace the target vector with the trial vector
                    self.X[j] = trial
                    self.Y[j] = obj_trial
            
            # Record the best individual of this generation
            generation_best_index = np.argmax(self.Y)
            current_best_y = self.Y[generation_best_index]
            
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(current_best_y)
            self.all_history_Y.append(self.Y.copy())
            
            print(f"Generation {i+1}: Best Accuracy = {current_best_y:.2f}%")
            
            # Update global best
            if current_best_y > self.best_y:
                 self.best_y = current_best_y
                 self.best_x = self.X[generation_best_index].copy()

        return self.best_x, self.best_y
