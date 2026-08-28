# Data and datasets

This folder holds the images the pipeline is evaluated on and the ground truth used to
score it. This document explains which datasets the project uses, where to get them, what
each one contains, and why it was chosen. The goal is that anyone reading this file can tell
what data the project runs on and understand the reasoning behind that choice.

## What the evaluation needs

The pipeline produces, for each input image, a stack of depth-ordered semantic RGBA layers.
Measuring quality therefore calls for three kinds of data:

1. A set of input images to run on, in a visual style that suits the pretrained backbones.
2. Ground-truth depth, to check that layers are ordered correctly from near to far.
3. Ground-truth object masks, to check that the semantic grouping is accurate.

Each dataset below covers one of these needs. The domain choice drives everything, so it is
described first.

## Domain choice: photorealistic everyday scenes (COCO-style)

The pipeline is built from pretrained experts. The segmentation model (Mask2Former) was
fitted on COCO panoptic categories, and the depth model (Depth Anything V2) was fitted on
large collections of natural photographs. Evaluation images are therefore drawn from
everyday photographic scenes, indoor and outdoor, containing the object classes the
grouping stage recognises (people, vehicles, furniture, animals). Matching the evaluation
domain to the backbones' training domain gives the fairest and most informative read on the
pipeline, because failures then reflect the decomposition logic rather than a domain gap.

The problem statement allows any style (photorealistic, anime, or vector). Photorealistic
everyday scenes are chosen because they are the domain the backbones handle most reliably
and the only one with public depth and mask ground truth at scale, which makes quantitative
scoring possible rather than purely visual.

## Datasets used

### 1. COCO 2017 (val split), for input images and segmentation ground truth
- Link: https://cocodataset.org/#download (2017 Val images and Panoptic annotations)
- Contents: about 5,000 validation images of common indoor and outdoor scenes, each with
  panoptic annotations that label every pixel as a countable object (person, car, chair) or
  a background region (sky, road, grass).
- Features that fit this project: the images contain exactly the classes the grouping stage
  targets, and the panoptic masks give per-instance ground truth for scoring segmentation
  mIoU. Because the segmenter was trained on these categories, COCO val is the natural test
  set for the semantic side of the deliverable.
- Role here: source of the frozen evaluation images in `test_images/`, and of the instance
  masks in `seg_masks/`.

### 2. DIODE, primary source of depth ground truth
- Link: https://diode-dataset.org/
- Contents: high-resolution RGB images paired with dense, accurate depth captured by a laser
  scanner, covering both indoor and outdoor scenes in one dataset.
- Features that fit this project: depth ordering is the geometric half of the deliverable,
  and DIODE is chosen as the primary depth source because it spans indoor and outdoor in a
  single, consistent capture. That range exercises the ordering logic across short indoor
  distances and long outdoor distances, which a purely indoor set would not. The depth is
  dense, so the per-layer depth statistic is stable.
- Role here: `diode/` depth maps, used to score depth-ordering accuracy.

### 3. NYU Depth v2, lighter indoor alternative for depth ground truth
- Link: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Contents: indoor RGB-D scenes (rooms, offices, homes) recorded with a Kinect sensor, with
  aligned depth maps.
- Features that fit this project: it is smaller, widely used, and quick to set up, which
  makes it a convenient indoor-only option when a full DIODE download is not needed. It is
  offered as a substitute for DIODE rather than in addition to it.
- Role here: optional `nyuv2/` depth maps, an alternative to `diode/`.

### 4. MULAN, optional layer-level ground truth
- Link: https://mulan-dataset.github.io/
- Contents: a dataset built specifically around multi-layer image decomposition, with
  annotations at the level of layers rather than single flat masks.
- Features that fit this project: it is the closest public match to the exact output of this
  pipeline, so it supports a more direct evaluation of the layer stack than depth or mask
  ground truth alone. It is listed as optional because its preparation tooling lives outside
  this repository.
- Role here: optional richer evaluation of the full layer stack.

## Summary of the choice

COCO val supplies the images and the segmentation ground truth because it matches the
classes and training domain of the segmenter. DIODE supplies depth ground truth because its
combined indoor and outdoor coverage tests ordering across the widest range of distances,
with NYU Depth v2 as a lighter indoor stand-in. MULAN is kept as an optional path to
layer-level scoring. Together these cover appearance, geometry, and semantics with public,
citable data in the domain the pipeline is strongest in.

## Folder layout

```
data/
├── test_images/     input images to run on (a frozen subset of COCO val)
├── seg_masks/       GT instance masks (from COCO panoptic) -> segmentation mIoU
├── diode/  or  nyuv2/   GT depth maps                      -> depth-ordering accuracy
├── failures/        hard or failed cases, for the limitations write-up
└── README.md        this file
```

### `test_images/`
A small frozen subset (about 20 to 50 images) of COCO val, kept fixed so reported numbers
stay honest. Accepts `.jpg`, `.jpeg`, `.png`. Used by:
```bash
python eval/run_benchmark.py --config configs/default.yaml \
    --images data/test_images --out out/bench
```

### `seg_masks/` (optional, enables segmentation mIoU)
One sub-folder per image, named by the image stem, holding one binary PNG per ground-truth
instance (white = object, black = background), at the image resolution:
```
data/seg_masks/street_01/person_1.png, car_1.png, ...
```
Enable with `--seg-gt data/seg_masks`. These are derived from COCO panoptic annotations.

### `diode/` or `nyuv2/` (optional, enables depth-ordering accuracy)
One depth map per image, named by the image stem:
- `<stem>.npy` (raw float depth, H x W), or
- `<stem>.png` (8 or 16-bit depth image).

Enable with `--depth-gt data/diode` (or `--depth-gt data/nyuv2`).

> Depth convention: the metrics assume smaller = nearer, matching the predicted layer depths.
> DIODE and NYU Depth v2 store metric distance (smaller = nearer), so no change is needed. If
> a source stores disparity or inverse depth (bigger = nearer), invert it before saving, or
> the ordering score will be reversed. The benchmark resizes ground truth to the image size
> with nearest-neighbour sampling and reads each layer with the same robust statistic used on
> the prediction (median over an eroded mask).

### `failures/`
A qualitative record of where the pipeline breaks, feeding the limitations section of the
report. Save the image (or the `--save-debug` visualization) plus a one-line note:
```
data/failures/crowd_overlap.png   note: two people merged into one layer
data/failures/glass_table.png     note: transparent surface mis-ordered
```

## Notes on obtaining the data

- Download COCO val images and panoptic annotations from the official site, then copy a
  representative subset into `test_images/` and convert the matching panoptic masks into the
  per-instance PNG layout above.
- Download DIODE or NYU Depth v2 separately and keep only the depth maps whose stems match
  your chosen `test_images/`.
- Model weights are not stored here; they download to the Hugging Face cache on first run.
