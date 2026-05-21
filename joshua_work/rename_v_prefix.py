from pathlib import Path


def rename_v_prefix(root_path: Path, dry_run: bool = False) -> int:
    """Rename files recursively under root_path by removing leading 'v_' from filenames."""
    renamed = 0
    root_path = root_path.expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    for path in root_path.rglob('*'):
        if path.is_file() and path.name.startswith('v_'):
            target = path.with_name(path.name[2:])
            if target.exists():
                print(f"Skipping existing target: {target}")
                continue
            print(f"Renaming: {path} -> {target}")
            if not dry_run:
                path.rename(target)
            renamed += 1
    return renamed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rename files by removing leading v_ from names.')

    # This is the line which defaults to my personal output
    parser.add_argument('root', nargs='?', default='joshua_work/output', help='Root folder to scan')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be renamed without changing files')
    args = parser.parse_args()

    root = Path(args.root)
    count = rename_v_prefix(root, dry_run=args.dry_run)
    print(f'Completed: {count} file(s) renamed under {root}')

# How to use:
# Run the script with the root directory as an argument. For example:
# & ".venv/Scripts/python.exe" "joshua_work/rename_v_prefix.py"