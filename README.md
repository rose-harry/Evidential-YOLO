# Evidential YOLO

- Code base for my MSc dissertation which focused on building uncertainty aware object detectors
- The motivation of the dissertation was to improve robustness of object detectors when dealing with out-of-distribution examples
- For example autonomous vehicles processing frames with intense sunlight or heavy rain
- Implementation is done by modifying the popular YOLO framework to enhance detections with real-time uncertainty estimates
- The proposed model for uncertainty estimation takes ideas from Subjective Logic and Evidential Deep Learning (EDL) and leads to Evidential YOLO, or EDL YOLO

### *Implementation and all modifications are built on-top of the open source YAD2K: Yet Another Darknet 2 Keras* - https://github.com/allanzelener/YAD2K

## Proposed Model
- Replace point estimates of normalised class probabilties with an alternative model
- Take ideas from subjective logic such that the sum of probabilties and uncertainty add to one
- Allows one to quantify epistemic uncertainty during inference

- For each object classification, treat class logits as evidence e such that e<sub>k</sub> is the generated evidence for class k. Applying the methodology, we map evidence e -> (p, u) where p is a probability vector and u is a scalar uncertainty estimate

<div>
<img src="/images/subjective_logic.png" width=700 >
</div>

#### This mapping applied to class logits defines EDL-YOLO
![Model]("/images/model.png")

## Comparing EDL-YOLO vs. YOLO v2.
### Out-of-distribution Examples
<div>
<img src="/images/cars_rain.png" title = "Rain" width=750>
<img src="/images/cars_sun.png" width=750>
</div>

### Rotation Invariance
- This example tests EDL-YOLO’s detection during rotation
- EDL-YOLO correctly detects both the upright and the rotated person despite rotating the image 90 degrees takes the image out-of-distribution
- The detection label is not only correct but generates increased uncertainty during rotation
<div>
<img src="/images/rotations.png" width=750>
</div>

## Modified Loss Function
<div>
<img src="/images/edl_loss_fcn.png" width=650>
</div>
 
## Training Curves
- Model learns to generate high amounts of evidence for true label when confident in prediction
- When model "doesn't know", loss function induces a uniform prior over class labels and hence zero evidence when uncertain
<div>
<img src="/images/training_curves.png" width=650>
</div>


