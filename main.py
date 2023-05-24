import pdb

from optimizer import *
from benchmark_funcs import *
from visualization import plot_optimization_history


def main(optimizer_type=adam, function=MullerBrownPotential):
    optimizer = Optimizer(optimizer_type, learning_rate=0.01)
    result = optimizer.optimize(function)
    plot_optimization_history(result, function)


if __name__ == '__main__':
    main()