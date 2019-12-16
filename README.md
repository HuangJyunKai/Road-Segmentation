# Road-Segmentation

## Download and Data format
1. Download dataset from http://www.cvlibs.net/datasets/kitti/eval_road.php or https://drive.google.com/file/d/1gakXvNJmnf9YPBt_taE_rlXduWtvMEqk/view?usp=sharing

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
4. Download weights file from https://drive.google.com/file/d/14bQbLzEjSucD7otv69iC5wCx36mVjDaF/view?usp=sharing
## Run script

```
Run convert.py
Run main.py train 
Run main.py check --ckpt=weights.pth  #predict image
```

## Requirement

```
python 3.6 up
pytorch (cuda supported)

```
