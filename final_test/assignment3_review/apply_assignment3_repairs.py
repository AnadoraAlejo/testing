#!/usr/bin/env python3
"""Conservative local-only Assignment 3 repairs.

Preserves all original starter methods and comments; makes backups before edits.
Does not commit or push to GitHub.
"""
from pathlib import Path
import argparse
import ast
import shutil
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', default='~/final_test',
                    help='ROS 2 workspace directory (default: ~/final_test)')
args = parser.parse_args()
workspace = Path(args.workspace).expanduser().resolve()
package = workspace / 'cave_explorer'
source_dir = package / 'cave_explorer'
reference = Path(__file__).with_name('advanced2a.py')

if not reference.is_file():
    raise SystemExit('Missing advanced2a.py alongside this script')
fixed = reference.read_text(encoding='utf-8')
ast.parse(fixed)
fix_path = source_dir / 'advanced2a.py'
if not fix_path.is_file():
    raise SystemExit(f'Missing source file: {fix_path}')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def backup_and_save(path, replacement):
    existing = path.read_text(encoding='utf-8')
    if existing == replacement:
        print('UNCHANGED:', path)
        return
    backup = path.with_name(path.name + '.backup_' + stamp)
    shutil.copy2(path, backup)
    ast.parse(replacement)
    path.write_text(replacement, encoding='utf-8')
    print('UPDATED: ', path)
    print('BACKUP:  ', backup)

# ONLY restore the known truncated 96-line Advanced 2A file.
current = fix_path.read_text(encoding='utf-8')
if current == fixed:
    print('Advanced 2A is already repaired')
elif current.splitlines() == fixed.splitlines()[:len(current.splitlines())] and current.rstrip().endswith(
        'def analyse(self, data, width, height, resolution, origin, robot_xy):'):
    backup_and_save(fix_path, fixed)
else:
    print('SKIPPED Advanced 2A: the file differs from the published truncated version.')
    print('Please compare it manually before replacing; no automatic overwrite occurred.')

# Only update two stale optional imports. Keep all other cave_explorer.py lines.
main = source_dir / 'cave_explorer.py'
if main.is_file():
    text = main.read_text(encoding='utf-8')
    text = text.replace('from cave_explorer.perception2 import ColourShapeDetector',
                        'from cave_explorer.perception2_detector import ColourShapeDetector')
    text = text.replace('from cave_explorer.perception3 import Perception3',
                        'from cave_explorer.perception3_localisation import Perception3')
    backup_and_save(main, text)

# Correct an outdated test import after the module rename.
tests = package / 'tests' / 'test_integrated_modules.py'
if tests.is_file():
    text = tests.read_text(encoding='utf-8')
    text = text.replace('from cave_explorer.advanced2_geometry import GeometryAnalyser',
                        'from cave_explorer.advanced2a import GeometryAnalyser')
    backup_and_save(tests, text)

# Check Python syntax of all package source and test modules.
errors = []
files = sorted(source_dir.glob('*.py')) + sorted((package / 'tests').glob('*.py'))
for path in files:
    try:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except SyntaxError as error:
        errors.append(f'{path}: {error}')
print('Syntax checked:', len(files), 'Python files')
if errors:
    print('Syntax errors:')
    for error in errors:
        print(' -', error)
    raise SystemExit(1)
print('Python source syntax: OK. ROS/Gazebo runtime remains untested.')
