"""Install the final experiment's method-specific source into an SDAR checkout."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdar-root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Copy files; otherwise show the plan')
    args = parser.parse_args()
    destination = args.sdar_root.resolve()
    if not (destination/'verl').is_dir() or not (destination/'agent_system').is_dir():
        parser.error('--sdar-root must point to an SDAR source checkout')
    source = ROOT/'overlay'
    paths = sorted(p for p in source.rglob('*') if p.is_file())
    for path in paths:
        relative = path.relative_to(source)
        target = destination/relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f'Target leaves the SDAR checkout: {relative}')
        if args.apply:
            backup = destination/'.c3-opd-overlay-backup'/relative
            if target.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        print(('Installed ' if args.apply else 'Would install ')+relative.as_posix())

if __name__ == '__main__':
    main()
