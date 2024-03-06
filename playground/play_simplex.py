import benchmark_functions as bf
import numpy as np

from NeuroNetworkByHands.Optimizers.simplex import Simplex

x0 = np.random.rand(3, 2)
f = bf.Ackley(n_dimensions=2)
f2 = bf.Rosenbrock(n_dimensions=2)
s = Simplex(f2)
trace = s.optimize(x0, trace=True)
f2.show(showPoints=trace)
