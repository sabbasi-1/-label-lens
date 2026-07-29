# YOLO Annotation Reviewer

A standalone, local desktop application for manually checking YOLO object-detection
annotations. It does not upload images or labels.

## Current capabilities

- Opens a YOLO `data.yaml` or a dataset directory.
- Discovers images and indexes their corresponding YOLO `.txt` files in the
  background with a visible, cancellable progress dialog.
- Filters the review queue to images containing any requested class IDs or names.
- Switches directly between Train, Validation, Test, and unspecified subsets.
- Jumps directly to any numeric position in the currently filtered queue.
- Draws labeled bounding boxes over each image.
- Reads mixed YOLO detection and segmentation-polygon label files.
- Draws polygon boundaries plus editable derived bounding rectangles without
  discarding the original mask points.
- Clicks a box to open a searchable class-selection dialog.
- Changes the selected box with single- or multi-digit class shortcuts.
- Draws missing boxes and moves, resizes, or deletes existing boxes.
- Keeps New Box mode active across repeated drawing and image navigation.
- Reuses the selected draw class for every following box until you change it.
- Replaces one annotation class with another in the current image, current split,
  current image folder, or images matching a filename pattern.
- Opens the current YOLO label file from a clickable path in the sidebar.
- Provides multi-step undo and redo for every annotation edit.
- Navigates with the left/right arrow keys.
- Offers optional auto-save when navigating or closing.
- Marks images as reviewed or flagged and resumes later.
- Filters the queue to unreviewed, flagged, or automatically suspicious images.
- Detects malformed rows, invalid normalized coordinates, degenerate boxes,
  unknown class IDs, duplicate boxes, high-overlap boxes, tiny or huge boxes,
  and extreme aspect ratios.
- Saves labels atomically and creates a one-time `.bak` copy of the original.
- Moves an image, its label, and its label backup to recoverable dataset-local
  trash when requested.
- Keeps review progress in `.yolo-review-state.json` at the dataset root.

The reviewer supports standard detection rows:

```text
class_id x_center y_center width height
```

It also supports standard YOLO segmentation rows:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Detection and polygon rows may be mixed in one file. A polygon is shown with its
boundary and a dashed derived bounding rectangle. Relabeling preserves all polygon
points. Moving or resizing the derived rectangle applies the same affine transform
to the polygon. Other unsupported row shapes, including oriented boxes, remain
blocked from saving so unknown data is not silently discarded.

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
| Choose dataset split | **Dataset split** selector |
| Jump to queue position | Enter a number and press Enter, or Ctrl+G |
| Select and relabel a box | Click the box |
| Search classes for selected box | Enter or Ctrl+L |
| Assign any class ID | Type its digits; pause briefly or press Enter |
| Move a box | Drag inside it |
| Resize a box | Drag one of its four corner handles |
| Select/edit mode | V or **Select / edit** |
| Persistent new-box mode | N or **New box**, then drag repeatedly |
| Change the active new-box class | C, **Draw class**, or type its ID |
| Adjust the newest box while drawing | Drag inside it or drag a corner |
| Draw deliberately over the newest box | Hold Shift while dragging |
| Leave new-box mode | V, **Select / edit**, or Escape |
| Delete selected box | Delete |
| Move current image and label to dataset trash | Ctrl+Delete or **Delete image + label** |
| Replace an annotation class | Ctrl+Shift+R or **Replace class...** |
| Save current labels | Ctrl+S |
| Toggle reviewed | R |
| Toggle flagged | F |
| Undo last relabel | Ctrl+Z |
| Redo edit | Ctrl+Y or Ctrl+Shift+Z |

Selecting overlapping boxes favors the smallest box under the pointer. The
annotation list on the right can always be used for exact selection. Double-click
an annotation in that list to search for a replacement class.

The current `.txt` annotation path appears in the sidebar. Click it to open the
file in the system's default text editor. If the image has unsaved annotation
changes, the reviewer saves them first so the opened file reflects the current
boxes and classes.

Every completed new box opens the class chooser with the active class already
selected. Accept another class to change the box and make that class active for
subsequent boxes, or cancel/close the chooser to keep the inherited class. While
**New box** remains active, the newest box can be moved or resized directly
without switching to Select/Edit mode.

## Bulk class replacement

Choose **Replace class...** or press `Ctrl+Shift+R`, then select the source class,
replacement class, and scope:

- **Current image** changes only the open image and creates one normal undo step.
- **Current split** targets every affected image in the current train, validation,
  test, or unspecified split.
- **Current image folder** targets images whose files share the current image
  directory.
- **Filename pattern** matches image filenames across the loaded dataset.

Filename patterns support `*` for any number of characters and `?` for one
character. For example, `camera_03_frame_*.jpg` targets that camera and numbering
scheme without relying on an ambiguous automatically inferred prefix. The dialog
prefills a pattern based on the current filename, which can be edited.

Before confirmation, the reviewer reports the number of images matched, label
files affected, and annotations that will change. Multi-file operations preflight
every affected label, save each file atomically, and roll written files back if
any save fails. They also preserve the exact pre-operation labels and a manifest
under:

```text
<dataset>/.label-lens-bulk-backups/<unique-id>/
```

Current-image replacement follows the usual manual/auto-save behavior. Larger
scopes are saved immediately after confirmation and clear the in-memory undo
history; use their operation backup for recovery after completion.

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

## Dataset splits and direct navigation

When `data.yaml` declares `train`, `val`/`valid`, and `test` sources, the sidebar
shows each available split with its image count. The split selector combines with
the review-status and class filters.

The jump field uses the current filtered queue. For example, after selecting
**Train (20,000)** and entering `1000`, the reviewer opens the 1,000th matching
training image. Press `Ctrl+G` to focus the field from the keyboard.

## Large datasets

Opening a dataset no longer blocks the application window. Image discovery is
shown as an indeterminate phase, followed by a determinate annotation-indexing
bar. The operation can be cancelled. Label parsing uses a bounded worker pool to
reduce startup time without creating one thread per image.

## Code structure

Label Lens is divided into UI-independent domain/services and Qt presentation
modules:

```text
src/yolo_reviewer/
|-- app.py                  # application bootstrap and compatibility exports
|-- core.py                 # backward-compatible facade for the old core API
|-- models.py               # annotation and dataset data types
|-- formats/
|   `-- yolo.py             # YOLO box/polygon parsing and path conventions
|-- services/
|   |-- dataset_loader.py   # YAML discovery, splits, and background index work
|   |-- review_session.py   # queue filtering and multi-image scope selection
|   |-- review_state.py     # reviewed/flagged persistence
|   |-- storage.py          # atomic saves, backups, rollback, and trash
|   `-- validation.py       # deterministic annotation checks
`-- ui/
    |-- canvas.py
    |-- dialogs.py
    |-- main_window.py
    `-- workers.py
```

The models, formats, and services do not depend on Qt widgets. Existing imports
from `yolo_reviewer.core` and `yolo_reviewer.app` remain supported through
compatibility exports.

## Data safety

By default, label files are changed only after an explicit save, navigation away
from a dirty image with confirmation, or application exit with confirmation.
Enable **Auto-save when changing images** to save without that prompt. Saving uses
a temporary file followed by an atomic replacement. Before the first replacement,
`<label>.txt.bak` is created. Review status is separate from annotations.

**Delete image + label** asks for confirmation, then moves the image, its label,
and any `.bak` label into `<dataset>/.label-lens-trash/<unique-id>/`, preserving
their relative paths. The item immediately leaves the active queue. This is
recoverable: move those files back to their original relative paths if needed.

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
