import numpy as np
from Diff_evo import DE

def test_func(x):
    return np.sum(x)

def run_test():
    n_dim = 5
    size_pop = 20
    max_iter = 50
    lb = [0] * n_dim
    ub = [10] * n_dim
    F = 0.5
    prob_mut = 0.7


    print("-+" * 25)
    print("Starting DE optimization")
    print("-+" * 25)
    de = DE(test_func, F, lb, ub, size_pop, n_dim, max_iter, prob_mut)
    best_x, best_y = de.run()

    print("-+" * 25)
    print(f"Best X: {best_x}")
    print(f"Best Y: {best_y}")
    print("-+" * 25)
    

if __name__ == "__main__":
    run_test()
