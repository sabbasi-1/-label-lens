# YOLO Annotation Reviewer

A standalone, local desktop application for manually checking YOLO object-detection
annotations. It does not upload images or labels.

## Current capabilities

- Opens a YOLO `data.yaml` or a dataset directory.
- Discovers images and indexes their corresponding YOLO `.txt` files in the
  background with a visible, cancellable progress dialog.
- Filters the review queue to images containing any requested class IDs or names.
- Draws labeled bounding boxes over each image.
- Clicks a box to open a searchable class-selection dialog.
- Changes the selected box with single- or multi-digit class shortcuts.
- Draws missing boxes and moves, resizes, or deletes existing boxes.
- Provides multi-step undo and redo for every annotation edit.
- Navigates with the left/right arrow keys.
- Offers optional auto-save when navigating or closing.
- Marks images as reviewed or flagged and resumes later.
- Filters the queue to unreviewed, flagged, or automatically suspicious images.
- Detects malformed rows, invalid normalized coordinates, degenerate boxes,
  unknown class IDs, duplicate boxes, high-overlap boxes, tiny or huge boxes,
  and extreme aspect ratios.
- Saves labels atomically and creates a one-time `.bak` copy of the original.
- Keeps review progress in `.yolo-review-state.json` at the dataset root.

The first version supports standard detection rows:

```text
class_id x_center y_center width height
```

Segmentation polygons and oriented bounding boxes are deliberately rejected rather
than being rewritten incorrectly. If such a row is present, saving that label file
is blocked and the original remains untouched.

## Install and run

From this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
yolo-reviewer
```

You can also launch it with:

```powershell
python -m yolo_reviewer
```

Choose either the dataset's `data.yaml` or its root directory. When a directory is
selected, the app looks for `data.yaml`, `dataset.yaml`, or `*.yaml`. If none is
present, it discovers images and uses numeric class names until names are supplied
by a YAML file.

## Expected layout

The conventional layout works without configuration:

```text
dataset/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

Paths declared by `train`, `val`, and `test` in `data.yaml` may be directories,
individual image files, lists, or text files containing image paths.

## Controls

| Action | Control |
|---|---|
| Previous / next image | Left / Right |
| Select and relabel a box | Click the box |
| Search classes for selected box | Enter or Ctrl+L |
| Assign any class ID | Type its digits; pause briefly or press Enter |
| Move a box | Drag inside it |
| Resize a box | Drag one of its four corner handles |
| Draw a missing box | N, then drag over the object |
| Delete selected box | Delete |
| Save current labels | Ctrl+S |
| Toggle reviewed | R |
| Toggle flagged | F |
| Undo last relabel | Ctrl+Z |
| Redo edit | Ctrl+Y or Ctrl+Shift+Z |

Selecting overlapping boxes favors the smallest box under the pointer. The
annotation list on the right can always be used for exact selection. Double-click
an annotation in that list to search for a replacement class.

## Filtering images by class

In the sidebar, enter one or more comma- or space-separated class IDs:

```text
0, 4, 11
```

Exact class names also work:

```text
helmet, vest
```

Press **Apply** or Enter. The resulting queue contains images with *any* of the
requested classes. This combines with the queue selector, so choosing
**Suspicious** plus classes `0, 4` shows suspicious images containing class 0 or
class 4. Press **Clear** to show every class again.

## Large datasets

Opening a dataset no longer blocks the application window. Image discovery is
shown as an indeterminate phase, followed by a determinate annotation-indexing
bar. The operation can be cancelled. Label parsing uses a bounded worker pool to
reduce startup time without creating one thread per image.

## Data safety

By default, label files are changed only after an explicit save, navigation away
from a dirty image with confirmation, or application exit with confirmation.
Enable **Auto-save when changing images** to save without that prompt. Saving uses
a temporary file followed by an atomic replacement. Before the first replacement,
`<label>.txt.bak` is created. Review status is separate from annotations.

## High-value automation roadmap

The built-in structural checks are deterministic and require no model. A later
model-assisted pass can rank images rather than silently changing annotations:

1. Run a trained YOLO checkpoint on every image.
2. Match predictions to annotations using IoU.
3. Rank strong class disagreements, missed objects, and low-confidence labels.
4. Review that ranked queue in this same interface.

Visual-embedding clustering is another useful pass: crop annotated objects, group
similar crops, and highlight labels that disagree with most neighbors. Both
methods should remain suggestions requiring human confirmation.
