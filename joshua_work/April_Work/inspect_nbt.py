import nbtlib
from pathlib import Path

dir_path = Path('joshua_work/air_clean_checker/martini2')
print('dir exists', dir_path.exists())
count = 0
for path in sorted(dir_path.rglob('*.nbt')):
    try:
        nbt = nbtlib.load(path)
    except Exception as e:
        print('ERR', path, e)
        continue
    has_palette = 'palette' in nbt
    has_blocks = 'blocks' in nbt
    if not has_palette or not has_blocks:
        print('NO palette/blocks', path, 'keys', list(nbt.keys()))
        count += 1
        continue
    blocks = nbt['blocks']
    if len(blocks) == 0:
        print('EMPTY blocks', path)
        count += 1
        continue
    only_air = True
    for block in blocks:
        state = block['state']
        name = nbt['palette'][state]['Name']
        if name != 'minecraft:air':
            only_air = False
            break
    if only_air:
        print('AIR ONLY', path, 'len', len(blocks))
        count += 1
print('candidates', count)
