import numpy as np

meta = []
# Define parameters for the random walk
dims = 3
step_n = 1200
step_set = [-1, -.5, -.3, 0, .3, .5, 1]
# Randomizing the velocity vector
sign = [-1, 1]
velocity_set=np.linspace(0.01,0.1,1000)
velocity = []
# Set the point of origin
origin = np.zeros((1, dims))
# Set up the size of the confinement
limits_set=np.linspace(0,6,1000)
limits=[]
# Simulate steps in 3D
for i in range(1500):
    # Create a random vector for the ballistic movement
    velocity = []
    velocity = np.random.choice(a=velocity_set, size=3)
    signs = np.random.choice(a=sign, size=3)
    velocity = np.multiply(velocity, signs)
    # velocity_z = np.sqrt(0.09 * 0.09 - velocity[0] * velocity[0] - velocity[1] * velocity[1]) * np.random.choice(a=sign,
    #                                                                                                              size=1)
    # velocity_z = np.sqrt(0.9 * 0.9 - velocity[0] * velocity[0] - velocity[1] * velocity[1]) * np.random.choice(a=sign,
    #                                                                                                              size=1)
    # velocity = np.concatenate([velocity, velocity_z])
    #
    # Define confinement
    limits=[]
    limits=np.random.choice(a=limits_set, size=6)
    limits[1]=limits[1]*(-1)
    limits[3]=limits[3]*(-1)
    limits[5]=limits[5]*(-1)
    # Create the diffusion component
    step_shape = (step_n, dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps])

    for k in range(1, len(path)):
        limits[0] = limits[0]+velocity[0]
        limits[1] = limits[1]+velocity[0]
        limits[2] = limits[2]+velocity[1]
        limits[3] = limits[3]+velocity[1]
        limits[4] = limits[4]+velocity[2]
        limits[5] = limits[5]+velocity[2]
        # if ((k % 1200 == 0)):
        #     limits = [1000,-1000,1000,-1000,1000,-1000]
        path[k] = path[k - 1] + path[k]
        while (path[k][0] > limits[0]) or (path[k][0] < limits[1]):
            if path[k][0] >= limits[0]:
                path[k][0] = 2*limits[0] - path[k][0]
            elif path[k][0] <= limits[1]:
                path[k][0] = 2*limits[1] - path[k][0]
        while (path[k][1] > limits[2]) or (path[k][1] < limits[3]):
            if path[k][1] >= limits[2]:
                path[k][1] = 2*limits[2] - path[k][1]
            elif path[k][1] <= limits[3]:
                path[k][1] = 2*limits[3] - path[k][1]
        while (path[k][2] > limits[4]) or (path[k][2] < limits[5]):
            if path[k][2] >= limits[4]:
                path[k][2] = 2*limits[4] - path[k][2]
            elif path[k][2] <= limits[5]:
                path[k][2] = 2*limits[5] - path[k][2]

    start = path[:1]
    stop = path[-1:]
    x, y, z, t = [], [], [], []


    for set in path:
        x.append(set[0])
        y.append(set[1])
        z.append(set[2])
    t = list(range(step_n + 1))
    meta.append(str(i)+": confinement "+str(limits)+", velocity "+str(velocity))

    fname = "ballistic and confined random\Limits_0-6au\Speed_0.01-0.1au/ballistic_confined_random_"+str(i)+".tck"
    f_out = open(fname, "w+")
    f_out.write('\t'.join(map(str, x))+'\n')
    f_out.write('\t'.join(map(str, y))+'\n')
    f_out.write('\t'.join(map(str, z))+'\n')
    f_out.write('\t'.join(map(str, t)))
    f_out.close()

f_out = open("ballistic and confined random\Limits_0-6au\Speed_0.01-0.1au/meta.txt", "w+")
for i in meta:
    f_out.write(i+'\n')
f_out.close()

