import numpy as np

meta = []
# Define parameters for the walk
dims = 3
step_n = 1200
step_set = [-1, -.5, -.3, 0, .3, .5, 1]
limits = [0.6, -0.6, 0.6, -0.6, 0.6, -0.6] # upper xlim, lower xlim, upper ylim, lower ylim, upper zlim, lower zlim
# limits = [1000, -1000, 1000, -1000, 1000, -1000]
origin = np.zeros((1, dims))

# Define the point of escape
escape = np.linspace(0.1,0.9,18)

# Simulate steps in 3D
for i in range(1500):
    step_shape = (step_n, dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps])
    point_of_escape = np.floor(np.random.choice(a=escape, size=1)*1500)
    for k in range(1, len(path)):
        # if ((k % 10 == 0) & (limits[0]<8)):
        # if (k % 10 == 0):
        #     limits = [x+.005 for x in limits]
        if ((k >= point_of_escape)):
            limits = [1000,-1000,1000,-1000,1000,-1000]
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
    limits = [.6, -.6, .6, -.6, .6, -.6]

    for set in path:
        x.append(set[0])
        y.append(set[1])
        z.append(set[2])
    t = list(range(step_n + 1))
    meta.append(str(i)+": escape at "+str(point_of_escape))

    fname = "confinement_escape/confinement_escape_"+str(i)+".tck"
    f_out = open(fname, "w+")
    f_out.write('\t'.join(map(str, x))+'\n')
    f_out.write('\t'.join(map(str, y))+'\n')
    f_out.write('\t'.join(map(str, z))+'\n')
    f_out.write('\t'.join(map(str, t)))
    f_out.close()

f_out = open("confinement_escape/meta.txt", "w+")
for i in meta:
    f_out.write(i+'\n')
f_out.close()
