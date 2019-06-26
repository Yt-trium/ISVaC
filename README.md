# ISVaC
Image Sensor Vacuum Cleaner  
A one night (+ one afternoon) project aiming to software remove dirt on my microscope image sensor.

## The idea
The idea was that, since the smudges always appear in the same places, it is possible to detect their position automatically (unsupervised learning) from a pool of images and a calculation of variance on pixels of the image.
Once the mask is calculated, it should be possible to reconstruct the pixels by interpolation of valid neighbors pixels. To treat the local areas independently of each other, we need to work by connected components.

## Results
The computation depend of two parameters :
* threshold : the threshold of variance to considere the pixel as a dirty pixel.
* dilatation : the number of time we "dilate" the mask to expand the area we will considere as dirty area.


### Masks (threshold / dilatation)

| 750 / 8  | 1000 / 4 | 1000 / 8 | 1100 / 4 |
|:--------:|:--------:|:--------:|:--------:|
| ![750_8_mask](results/results_750_8/MASK.JPG "750_8_mask") | ![1000_4_mask](results/results_1000_4/MASK.JPG "1000_4_mask") | ![1000_8_mask](results/results_1000_8/MASK.JPG "1000_8_mask") | ![1100_4_mask](results/results_1100_4/MASK.JPG "1100_4_mask") |


### Example of results with the 4 examples configurations

| Before        | After         | Mask red highlight |
|:-------------:|:-------------:|:-------------:|
| ![PICT0037](data/PICT0037.JPG "PICT0037") | ![PICT0037_750_8](results/results_750_8/PICT0037.JPG "PICT0037_750_8") | ![PICT0051_750_8](results/masks_750_8/PICT0037.JPG "PICT0037_750_8") |
| ![PICT0042](data/PICT0042.JPG "PICT0042") | ![PICT0042_1000_4](results/results_1000_4/PICT0042.JPG "PICT0042_1000_4") | ![PICT0042_1000_4](results/masks_1000_4/PICT0042.JPG "PICT0042_1000_4") |
| ![PICT0052](data/PICT0052.JPG "PICT0052") | ![PICT0052_1000_8](results/results_1000_8/PICT0052.JPG "PICT0052_1000_8") | ![PICT0052_1000_8](results/masks_1000_8/PICT0052.JPG "PICT0052_1000_8") |
| ![PICT0051](data/PICT0051.JPG "PICT0051") | ![PICT0051_1100_4](results/results_1100_4/PICT0051.JPG "PICT0051_1100_4") | ![PICT0051_1100_4](results/masks_1100_4/PICT0051.JPG "PICT0051_1100_4") |
