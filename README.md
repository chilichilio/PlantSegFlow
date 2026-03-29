## PlantSegFlow
Depth-Guided SAM3 Plant Segmentation & Row-Level Tracking
## Overview
PlantSegFlow is a depth-guided zero-shot segmentation and tracking pipeline for extracting plant-level phenotypes from UAV and ground RGB videos.

## Demo on Lantana trial
![Demo](/final_seg_res_example.gif)

## Environment Setup (SAM3)
This project relies on SAM3. 
Please follow the official installation and environment instructions from the SAM3 repository:
## SAM3 Official Repository (Installation Guide) 
https://github.com/facebookresearch/sam3
Follow their environment setup steps (Python, PyTorch, CUDA, checkpoint download) before running this pipeline.
## Dataset
RGB + depth visualization videos: Dropbox Dataset link: https://www.dropbox.com/scl/fo/96710jijok37z8vz6wajx/AMzqwtNZTYnJfSgfkuqX6Wg?rlkey=gyq2bfnxyvwi4fjyqijan9bot&st=1d0ygik3&dl=0

Each sequence folder must contain paired files:

```text
xxx_src.mp4   # Colored RGB video (output from Video Depth Anything)
xxx_vis.mp4   # Depth visualization video (output from Video Depth Anything)
```
## Run Plant Canopy Segment Algorithm
Place ```VideoCanopySegment.py``` inside the SAM3 project folder (same level as the SAM3 repository root).
```text
python VideoCanopySegment.py --seq_dir /path/to/sequence --fps 15 --prompt "plant"
```

## Output
```text
sequence_segment/
│
├── components_topk/                      # Top-K canopy component masks
├── masks_union/                          # Union masks (SAM3 + depth)
├── depth_keep_masks/                     # Binary depth masks
├── measurements_topk.csv                 # Per-frame component measurements
├── timing/                               # Stage timing diagnostics
│
└── hitprocess/
    │
    ├── hit_center_with_xy_area.csv       # Final tracking result
    ├── hit_masks/                        # Selected masks after circle-hit
    ├── components_topk_with_two_centers/ # Visualization with row centers
    ├── largest_per_plant/                # Plant-level cropped RGB outputs
    └── kmeans_time_vs_x_k2_circleR50.png # Row clustering plot
```
## Key Parameters | Parameter | Description | 
| Parameter  | Description           |
| ---------- | --------------------- |
| `--fps`    | Frame extraction rate |
| `--prompt` | SAM3 text prompt      |
| `--k_rows` | Number of crop rows   |


## SAM3 checkpoint expected at:

sam3/checkpoints/sam3.pt

## To customise your own video
## Generate Depth Video (Video Depth Anything) 
For monocular depth estimation from RGB videos:

https://github.com/DepthAnything/Video-Depth-Anything

After installing Video Depth Anything, you can generate the depth visualization video using:
```text
python FolderDepthAnything.py --video your_video.mp4
```
## Output
```text
xxx_src.mp4   # Colored RGB video (output from Video Depth Anything)
xxx_vis.mp4   # Depth visualization video (output from Video Depth Anything)
```
Then run:
```text
python VideoCanopySegment.py --seq_dir /path/to/your_sequence 
```
