# Evidential-YOLO

### Code base for my MSc dissertation which focused on building uncertainty aware object detectors. The motivation for the dissertation was to improve the robustness of object detectors when dealing with out-of-distribution examples, for example autonomous vehicles processing frames with intense sunlight or heavy rain. 

### The implementation is done by modifying the popular YOLO framework to enhance detections with real-time uncertainty estimates. The base model relies heavily on YAD2K: Yet Another Darknet 2 Keras:

## https://github.com/allanzelener/YAD2K


![Model](/images/model.png)

## EDL-YOLO vs. YOLO v2.
### Out-of-distribution Examples
<div>
<img src="/images/cars_rain.png" title = "Rain" width=750>
<img src="/images/cars_sun.png" width=750>
</div>

### Rotation Invariance
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


