import numpy as np
from pathlib import Path

p = Path('joshua_work/cell_input/martini2_cell.npy')
cell_data = np.load(p)
arr = cell_data[0]
print('shape:', arr.shape)
for i in range(arr.shape[-1]):
    count = int(np.sum(arr[..., i]))
    print(f'channel {i}: {count}')
