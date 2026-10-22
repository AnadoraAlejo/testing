#!/usr/bin/env python3
"""Human box annotation for the supplied stop-sign model (desktop OpenCV)."""
import argparse
import json
from pathlib import Path
import os
import cv2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image_directory')
    parser.add_argument('--output',default='evaluation_labels.json')
    args=parser.parse_args()
    root=Path(args.image_directory).expanduser().resolve()
    output=Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new output filename')
    paths=sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    if not paths:
        raise ValueError('No images found')
    records=[]
    print('Draw every stop sign. ENTER/SPACE accepts a box; ESC ends this image.')
    print('No stop signs: press ESC without drawing. Inspect negatives carefully.')
    for path in paths:
        image=cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot read '+str(path))
        print('Label:',path)
        boxes=cv2.selectROIs('Label all stop signs; ESC finishes image',image,showCrosshair=True,fromCenter=False)
        records.append(dict(image=os.path.relpath(path,output.parent),
                            objects=[dict(label='stop_sign',box=list(map(int,box))) for box in boxes if box[2]>0 and box[3]>0]))
        # Checkpoint after every image. Do not overwrite an existing run at startup.
        output.write_text(json.dumps(records,indent=2))
    cv2.destroyAllWindows()
    print('Saved',output)


if __name__=='__main__':
    main()
