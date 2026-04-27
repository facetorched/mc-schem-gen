import numpy as np
cell_data = np.load('joshua_work/cell_input/martini2_cell.npy')
print('Checking different channels:')
for i in range(6):
    voxels = cell_data[0, 100:150, 100:150, 100:150, i]
    print(f'Channel {i}: {np.sum(voxels)} True voxels')