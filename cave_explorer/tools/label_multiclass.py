#!/usr/bin/env python3
# === ASSIGNMENT 3 PROVENANCE (LOCAL ANNOTATION) ===
# Source branch / provenance: LOCAL INTEGRATION — not from a GitHub branch
# GitHub commit account / updater: ChatGPT-assisted integration; adopted/tested by user locally
# Pinned commit SHA: none — new local integration
# Integration notes: New multiclass bounding-box labelling helper for assessment metrics.
# Attribution caveat: GitHub account != original author of every line; disclose AI assistance.
# === END ASSIGNMENT 3 PROVENANCE ===
"""Create reviewed multi-class annotation manifest for evaluator.

Draw boxes on each image, then type a class label for each box in terminal.
Use only on images you have rights to label. ESC with no boxes = negative.
"""
import argparse
import json
import os
from pathlib import Path
import cv2

CLASSES = ('stop_sign', 'green_alien', 'green_crystals',
           'white_sphere', 'ice_formation', 'blue_mushrooms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_directory')
    parser.add_argument('--output', default='evaluation_multiclass.json')
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit('Refusing to overwrite existing annotation file')
    root = Path(args.image_directory).expanduser().resolve()
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not paths:
        raise SystemExit('No images found')
    print('Labels:', ', '.join(CLASSES))
    print('Mark EVERY artefact in each image. ENTER after boxes; ESC for no more.')
    annotations = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Unreadable image ' + str(path))
        boxes = cv2.selectROIs('Select ALL artefacts: ' + path.name, image,
                               showCrosshair=True, fromCenter=False)
        objects = []
        for box in boxes:
            if int(box[2]) <= 0 or int(box[3]) <= 0:
                continue
            while True:
                label = input(f'{path.name} box {list(map(int,box))} class: ').strip().lower()
                if label in CLASSES:
                    break
                print('Type one of:', ', '.join(CLASSES))
            objects.append(dict(label=label, box=list(map(int, box))))
        annotations.append(dict(image=os.path.relpath(path, output.parent), objects=objects))
        output.write_text(json.dumps(annotations, indent=2))
        print('Checkpoint:', len(annotations), '/', len(paths))
    cv2.destroyAllWindows()
    print('Complete:', output)


if __name__ == '__main__':
    main()
