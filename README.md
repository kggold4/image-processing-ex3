# Image Processing and Computer Vision Assignment 3
### <i>Pyramids and Optic Flow</i>

## Util functions list:
> help functions:
1. myID
2. get_base_kernel
3. get_sigma
4. get_list_t
5. double_win
6. pyr_blend_help

> assignment functions:
1. opticalFlow
2. opticalFlowPyrLK
3. findTranslationLK
4. findRigidLK
5. findTranslationCorr
6. findRigidCorr
7. warpImages
8. gaussianPyr
9. blurImage2
10. laplaceianReduce
11. get_gaussian_kernel
12. gaussExpand
13. laplaceianExpand
14. pyrBlend

> My main file output (terminal):

```
LK Demo
Time: 0.0898
[-0.07858644 -0.03752994]
[-0.07055907 -0.03504174]
Hierarchical LK Demo
Time: 1.2606
[ 0.48307252 -0.46760026]
[ 0.94970752 -1.03342365]
Compare LK & Hierarchical LK
Compare LK & Hierarchical LK
Time of naive method: 0.1576
median of naive method: [ 0.28846266 -0.3836858 ]
mean of naive method: [ 0.383214   -0.47046884]
Time of hierarchical method: 1.1788
median of hierarchical method: [ 0.48771212 -0.45579601]
mean of hierarchical method: [ 0.96132863 -1.02765762]
accuracy improved by: 0.7064851810521642
Image Warping Demo
Compare LK & Hierarchical LK (Movement)
Original transformation:
[[ 1.  0. -5.]
 [ 0.  1.  8.]
 [ 0.  0.  1.]]
-LK Results:
Time: 0.0309
Translation found:
[[ 1.         0.        -0.4689346]
 [ 0.         1.         6.14677  ]
 [ 0.         0.         1.       ]]
SE:
-Correlation Results:
Time: 1.1220
Translation found:
[[ 1.  0. -2.]
 [ 0.  1. 11.]
 [ 0.  0.  1.]]
SE:
Compare LK & Hierarchical LK (Rigid)
Original transformation:
[[  0.9396926   -0.34202015  12.        ]
 [  0.34202015   0.9396926  -12.5       ]]
-LK Results:
Time: 0.0688
Translation found:
[[  0.65362081  -0.7568222    2.61424231]
 [  0.7568222    0.65362081 -10.49939242]
 [  0.           0.           1.        ]]
SE:
-Correlation Results:
Time: 1.2237
Translation found:
[[ 0.89442719 -0.4472136   8.        ]
 [ 0.4472136   0.89442719 16.        ]
 [ 0.          0.          1.        ]]
SE:
Gaussian Pyramid Demo
Laplacian Pyramid Demo
```

<br>

> My machine system details:

* RAM: ```16GB```
* MAIN ROM: ```512GB SSD```
* CPU: ```intel i7-8750H, 12 logic CPUs (6 physical)```
* GPU: ```Nvidia GeForce GTX 1070 8GB```
