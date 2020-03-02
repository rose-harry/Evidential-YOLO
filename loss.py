import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, Dropout, Dense, Activation, LeakyReLU
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from edl_yolo import (preprocess_true_boxes, yolo_body, yolo_eval, yolo_head)

def custom_loss(args,
                anchors,
                num_classes,
                global_step = 0.,
                rescore_confidence=False):
    """
    Modified YOLO localization loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

    detectors_mask : array
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : array
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.

    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 2.5
    coordinates_scale = 1
    edl_scale = 2.5

    yad2kOutput, edlOutput = yolo_head(yolo_output, anchors, num_classes, clip = 5.)
    pred_xy, pred_wh, pred_confidence, pred_softmax_class_probs = yad2kOutput
    pred_class_logits, pred_box_class_evidence, pred_alpha, \
                pred_S, pred_uncertainty, pred_class_probs = edlOutput

    # Unadjusted box predictions for loss.
    # TODO: Remove extra computation shared with yolo_head.
    yolo_output_shape = K.shape(yolo_output)
    feats = K.reshape(yolo_output, [
        -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
        num_classes + 5
    ])
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

    # TODO: Adjust predictions by image width/height for non-square images?
    # IOUs may be off due to different aspect ratio.

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = K.shape(true_boxes)

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLO does not use binary cross-entropy here.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * K.square(-pred_confidence)

    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        K.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)
    classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_probs))

    # Coordinate loss for matching detection boxes.
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))


    ########################################################
    ########################################################
    ########                                       #########
    ######## EDL Loss and metric calculations here #########
    ########                                       #########
    ########################################################
    ########################################################
    
    ########             EDL Loss                  #########

    ### EDL Loss - expected value of cross entropy loss over
    # the predicted Dirichlet distribution + KL regularization term

    # Expected value of cross entropy loss
    A = tf.reduce_sum(matching_classes * (tf.digamma(pred_S) - tf.digamma(pred_alpha)), 4, keepdims=True)

    # KL term
    alp = pred_box_class_evidence * (1-matching_classes) + 1
    beta = K.ones_like(alp)
    S_alpha = tf.reduce_sum(alp,axis=4,keep_dims=True)
    S_beta = tf.reduce_sum(beta,axis=4,keep_dims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alp),axis=4,keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta),axis=4,keep_dims=True) - tf.lgamma(S_beta)
    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alp)
    kl = tf.reduce_sum((alp - beta)*(dg1-dg0),axis=4,keep_dims=True) + lnB + lnB_uni
    
    #annealing_coeff = 2.0 * tf.minimum(1.0,  tf.cast(global_step / annealing_step, tf.float32))
    annealing_coeff = 5.0
    B = annealing_coeff * kl     # Anneal the KL term during training phase

    # 5. Apply detector mask and sum the loss components
    edl_loss =  edl_scale * detectors_mask * (A + B)
    
    
    # EDL loss components
    exp_ce_loss_sum = tf.reduce_sum( detectors_mask * A )
    kl_loss_sum     = tf.reduce_sum( detectors_mask * kl )
    akl_loss_sum    = annealing_coeff * kl_loss_sum

    ########             EDL Metrics               #########
    
    preds = tf.cast(tf.argmax(pred_box_class_evidence, 4), 'int32')
    truth = tf.cast(matching_true_boxes[..., 4], 'int32')
    matchs = tf.cast(tf.equal(preds, truth), tf.float32)
    match = tf.boolean_mask( tf.expand_dims(matchs,4), detectors_mask)
    acc = tf.reduce_mean(match)
    
    total_evidence = tf.reduce_sum(pred_box_class_evidence, 4, keepdims=True)
    total_evidence = tf.boolean_mask(total_evidence, detectors_mask)
    
    mean_ev_succ =  tf.reduce_sum(total_evidence*match) / tf.reduce_sum(match+1e-20)
    mean_ev_fail =  tf.reduce_sum(total_evidence*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20)
     
    ########################################################
    ########################################################    
    

    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    edl_loss_sum =  K.sum(edl_loss)

    total_loss = 0.5 * ( confidence_loss_sum + edl_loss_sum + coordinates_loss_sum + classification_loss_sum)
   
    return tf.stack([total_loss, confidence_loss_sum, classification_loss_sum, edl_loss_sum, coordinates_loss_sum, 
                      acc, mean_ev_succ, mean_ev_fail, 
                      annealing_coeff, exp_ce_loss_sum, kl_loss_sum, akl_loss_sum])