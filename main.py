import pdb

from optimizer import *
from benchmark_funcs import *
from visualization import plot_optimization_history


def main(optimizer_type=newton_method, second_optimizer_type=adam, function=MullerBrownPotential):
    optimizer = Optimizer(optimizer_type, second_optimizer_type, learning_rate=0.01)
    history = optimizer.optimize(function)
    plot_optimization_history(history, function)


if __name__ == '__main__':
    main()