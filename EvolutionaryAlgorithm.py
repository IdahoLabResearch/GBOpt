import numpy as np
import matplotlib.pyplot as plt
from GBMaker import GBMaker
import math
import subprocess

def evolve(G,GBE):
    sort_idx = np.argsort(GBE)

    sorted_batch = initial_batch[sort_idx]

    S_60 = int(initial_batch_size*0.6)
    S_10 = int(initial_batch_size*0.1)

    parents = sorted_batch[:S_60]

    top_10 = sorted_batch[:S_10]

    S_50 = S_60-S_10

    slice_thickness = G.gb_thickness*np.random.rand(S_50)
    gb_struct_pairs = np.random.randint(0,S_60,size=(S_50,2))

    GBE = [GBE[i] for i in sort_idx]
    for (i,A) in enumerate(zip(slice_thickness,gb_struct_pairs)):
        t,gb_pair = A

        print("Cut line x = ",G.grain_xdim-G.gb_thickness/2 +t)

        G.translate_right_grain(dy=parents[gb_pair[0] ][0],dz=parents[gb_pair[0] ][1])
        pos1 = G.positions
        pos1 = pos1[np.where(pos1[:,0] <= G.grain_xdim-G.gb_thickness/2 +t)]

        G.translate_right_grain(dy=parents[gb_pair[1] ][0],dz=parents[gb_pair[1] ][1])
        pos2 = G.positions
        pos2 = pos2[np.where(pos2[:,0] >= G.grain_xdim-G.gb_thickness/2+t)]

        new_positions = np.vstack([pos1,pos2])
        G.positions = new_positions
        G.write_lammps('gb_struct.data',autogen_positions=False)
        subprocess.run('mpirun -n 40 lmp -in lmp_uspex.test > output.txt', shell=True)
        with open('results.out', 'r') as file:
            txt = file.read()
            gbe_val = (float(txt.split(' ')[0]))
        print(gbe_val)

        GBE+=[gbe_val]
    return GBE

initial_batch_size = 10 # How many samples to start the evolutionary algo with
N = 2                   # How many parameters are we optimizing over?


"""
Create grain boundary structure
"""
theta = math.radians(5)
gb_thickness=10
G = GBMaker(lattice_parameter=3.61, gb_thickness=gb_thickness,
            misorientation=[theta, 0, 0], repeat_factor=10)

max_dy = G.spacing['y']
max_dz = G.spacing['z']

max_arr = np.asarray([max_dy,max_dz])

random = np.random.rand(initial_batch_size,N)

initial_batch = max_arr*random

GBE = []
for batch in initial_batch:
    G.translate_right_grain(dy=batch[0],dz=batch[1])
    G.write_lammps('gb_struct.data')
    # This will be replaced with a batch submission of PBS jobs in the future
    subprocess.run('mpirun -n 40 lmp -in lmp_uspex.test > output.txt', shell=True)
    with open('results.out', 'r') as file:
        txt = file.read()
        gbe_val = (float(txt.split(' ')[0]))
    GBE+=[gbe_val]

batch_min_gbe = min(GBE)

GBE = evolve(G,GBE)
print(GBE)
