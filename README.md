# Evidential YOLO

- Code base for my MSc dissertation which focused on building uncertainty aware object detectors
- The motivation of the dissertation was to improve robustness of object detectors when dealing with out-of-distribution examples
- For example autonomous vehicles processing frames with intense sunlight or heavy rain
- Implementation is done by modifying the popular YOLO framework to enhance detections with real-time uncertainty estimates

### Implementation and all modifications are built on-top of the open source YAD2K: Yet Another Darknet 2 Keras - https://github.com/allanzelener/YAD2K

## Proposed Model
- Replace point estimates of normalised class probabilties with an alternative model
- Take ideas from subjective logic such that the sum of probabilties and uncertainty add to one
- Allows one to quantify epistemic uncertainty during inference

More formally, defining the Dirichlet strength S = &sum;<sub>k</sub> e<sub>k</sub> + 1, we see given evidence e<sub>k</sub> &ge; 0 for the kth class, the belief mass b<sub>k</sub> and uncertainty u are computed by

<a href="https://www.codecogs.com/eqnedit.php?latex=$$b_k&space;=&space;\frac{e_k}{S}&space;\text{&space;and&space;}&space;u&space;=&space;\frac{K}{S}.$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$b_k&space;=&space;\frac{e_k}{S}&space;\text{&space;and&space;}&space;u&space;=&space;\frac{K}{S}.$$" title="$$b_k = \frac{e_k}{S} \text{ and } u = \frac{K}{S}.$$" /></a>

Such that all K+1 parameters are non-negative and are additive, i.e., 

<a href="https://www.codecogs.com/eqnedit.php?latex=$$u&space;&plus;&space;\sum_{k=1}^K&space;b_k&space;=&space;1.&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$u&space;&plus;&space;\sum_{k=1}^K&space;b_k&space;=&space;1.&space;$$" title="$$u + \sum_{k=1}^K b_k = 1. $$" /></a>

- For each object classification, treat logits as evidence e such that e<sub>k</sub> is generated evidence for class k.

![Model](/images/model.png)

## EDL-YOLO vs. YOLO v2.
### Out-of-distribution Examples
<div>
<img src="/images/cars_rain.png" title = "Rain" width=750>
<img src="/images/cars_sun.png" width=750>
</div>

### Rotation Invariance
- 
<div>
<img src="/images/rotations.png" width=750>
</div>

## Modified Loss Function
<div>
<img src="/images/edl_loss_fcn.png" width=750>
</div>
 
## Training Curves
<div>
<img src="/images/training_curves.png" width=750>
</div>


