import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Define parameters for the walk
dims = 3
step_n = 1200
step_set = [-1, -.5, -.3, 0, .3, .5, 1]
origin = np.zeros((1, dims))
# Simulate steps in 3D
for i in range (50):
    step_shape = (step_n, dims)
    steps = np.random.choice(a=step_set, size=step_shape) # random choice of a step size in x, y, z (3 separate numbers) out of the step_set array
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]
    x, y, z, t = [], [], [], []


    for set in path:
        x.append(set[0])
        y.append(set[1])
        z.append(set[2])
    t = list(range(step_n+1))

    fname = "random_test_"+str(i)+".tck"
    f_out = open(fname, "w+")
    f_out.write('\t'.join(map(str, x))+'\n')
    f_out.write('\t'.join(map(str, y))+'\n')
    f_out.write('\t'.join(map(str, z))+'\n')
    f_out.write('\t'.join(map(str, t)))
    f_out.close()

