import numpy as np

# Define parameters for the random walk
dims = 3
step_n = 1200
step_set = [-1, -.5, -.3, 0, .3, .5, 1]
# Randomizing the velocity vector
sign = [-1, 1]
velocity_set=np.linspace(-0.5,0.5,1000)
print(velocity_set)
# velocity_set = [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
# turn on the "fast" ballistic movement
# velocity_set = velocity_set*10
velocity = []
# Set the point of origin
origin = np.zeros((1, dims))
# Simulate steps in 3D
for i in range(100):
    # Create a random vector for the ballistic movement
    velocity = []
    velocity = np.random.choice(a=velocity_set, size=3)
    # velocity_z = np.sqrt(0.09 * 0.09 - velocity[0] * velocity[0] - velocity[1] * velocity[1]) * np.random.choice(a=sign,
    #                                                                                                              size=1)
    # velocity_z = np.sqrt(0.9 * 0.9 - velocity[0] * velocity[0] - velocity[1] * velocity[1]) * np.random.choice(a=sign,
    #                                                                                                              size=1)
    # velocity = np.concatenate([velocity, velocity_z])
    # Create the diffusion component
    step_shape = (step_n, dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    steps = steps + velocity
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]
    x, y, z, t = [], [], [], []


    for set in path:
        x.append(set[0])
        y.append(set[1])
        z.append(set[2])
    t = list(range(step_n + 1))

    fname = "ballistic and simple diffusion random/ballistic_simple_random_"+str(i)+".tck"
    f_out = open(fname, "w+")
    f_out.write('\t'.join(map(str, x))+'\n')
    f_out.write('\t'.join(map(str, y))+'\n')
    f_out.write('\t'.join(map(str, z))+'\n')
    f_out.write('\t'.join(map(str, t)))
    f_out.close()

