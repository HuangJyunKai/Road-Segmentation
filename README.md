# Road-Segmentation

## Download and Data format
1. Download dataset from http://www.cvlibs.net/datasets/kitti/eval_road.php

2. Unzip zip file

3. Rename the data folder 
```
-- data_road\ 
    -- training\
       -- image\
          -- um_000000.png
          -- um_000001.png
          ...
       -- label\
          -- um_lane_000000.png
          -- um_lane_000001.png
          ...
       -- label_grayscale\
          ...
       -- calib\
          ...
    -- testing\
       -- image\
          -- um_000000.png
       -- calib\
          ...
```
## Run script

```
Run convert.py
Run main.py train 
```

## Requirement

```
python 3.6 up
pytorch (cuda supported)

```
