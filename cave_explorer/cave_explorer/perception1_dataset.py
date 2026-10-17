from pathlib import Path

import cv2

# names of the stuff in the cave
# negative means the photo has no artefact in it
KNOWN_LABELS = (
    'green_alien',
    'green_crystals',
    'white_sphere',
    'ice_formation',
    'blue_mushrooms',
    'stop_sign',
    'negative',
)


def clean_label(label):
    # turns whatever we type into a folder name
    # so spaces and capital letters dont break the save
    if label is None:
        return 'negative'
    cleaned = ''.join(
        character if character.isalnum() else '_'
        for character in str(label).strip().lower()
    )
    cleaned = '_'.join(part for part in cleaned.split('_') if part)
    return cleaned or 'negative'


def to_bgr(image, encoding):
    # gazebo sends the picture one way and opencv saves it the other way
    # so i swap the colours or the photos come out wrong
    # empty picture just gets skipped
    if image is None or image.size == 0:
        return None
    enc = (encoding or '').lower()
    if enc in ('rgb8', 'rgb'):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if enc == 'rgba8':
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def prepare_dataset(root):
    # makes a folder for each artefact
    # also counts photos already there so we dont overwrite them
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for label in KNOWN_LABELS:
        folder = root / label
        folder.mkdir(exist_ok=True)
        counts[label] = len(list(folder.glob(f'{label}_*.jpg')))
    return root, counts


def save_frame(root, counts, image_bgr, label):
    # writes one photo and updates the count list
    # counts.txt is what we use in the report
    label = clean_label(label)
    folder = Path(root) / label
    folder.mkdir(parents=True, exist_ok=True)
    counts[label] = counts.get(label, 0) + 1
    path = folder / f'{label}_{counts[label]:04d}.jpg'
    if not cv2.imwrite(str(path), image_bgr):
        counts[label] -= 1
        raise RuntimeError(f'Could not write {path}')
    lines = []
    names = list(KNOWN_LABELS)
    for name in counts:
        if name not in names:
            names.append(name)
    for name in names:
        lines.append(f'{name}: {counts.get(name, 0)}')
    (Path(root) / 'counts.txt').write_text('\n'.join(lines) + '\n')
    return path

# PERCEPTION 1 END
