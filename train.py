import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pickle
import io
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, Dropout, Dense, Activation, LeakyReLU
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from helpers import resize, orig_size, get_detector_mask, process_boxes, shuffle_data, batch_generator


def create_model(anchors, class_names, load_pretrained=True, freeze_body=True, reset_weights=True):
    '''
    returns the body of the model and the model
    # Params:
    load_pretrained: whether or not to load the pretrained model or initialize all weights
    freeze_body: whether or not to freeze all weights except for the last layer's
    # Returns:
    model_body: YOLOv2 with new output layer
    model: YOLOv2 with custom loss Lambda layer
    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('EDLyolo/model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('EDLyolo/model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

  
    # Reset Weights
    if reset_weights:
      sess = K.get_session()
      initial_weights = topless_yolo.get_weights()
      from keras.initializers import glorot_uniform  # Or your initializer of choice
      k = 65
      new_weights = [glorot_uniform()(initial_weights[i].shape).eval(session=sess) if i>k else initial_weights[i] for i in range(len(initial_weights))]
      topless_yolo.set_weights(new_weights)

    if freeze_body:
      for layer in topless_yolo.layers[:-30]:
        layer.trainable = False

    output_size = len(anchors)*(5+len(class_names))
    
    final_layer = Conv2D( output_size , (1, 1), activation='linear')( topless_yolo.output )
       
    model_body = Model(image_input, final_layer)
 
    with tf.device('/cpu:0'):
      global_step = K.variable(0.)
      model_metrics = Lambda(
          custom_loss,
          output_shape=(12, ),
          name='yolo_metrics',
          arguments={'anchors': anchors,
                     'num_classes': len(class_names),
                      'global_step': global_step})([
                         model_body.output, boxes_input,
                         detectors_mask_input, matching_boxes_input
                     ])

      # Total Loss Components
      total_loss = Lambda(lambda x: x[0], name='yolo_loss')( model_metrics )
      loss_conf  = Lambda(lambda x: x[1], name='loss_conf')( model_metrics )
      loss_class = Lambda(lambda x: x[2], name='loss_class')( model_metrics )
      loss_edl   = Lambda(lambda x: x[3], name='loss_edl')( model_metrics )
      loss_coord = Lambda(lambda x: x[4], name='loss_coord') ( model_metrics )

      # EDL Metrics
      accrcy     = Lambda(lambda x: x[5], name='accrcy')( model_metrics )
      ev_succ    = Lambda(lambda x: x[6], name='ev_succ')( model_metrics )  
      ev_fail    = Lambda(lambda x: x[7], name='ev_fail')( model_metrics )  

      # EDL Loss Components
      annealing_coeff  = Lambda(lambda x: x[8], name='annealing_coeff')( model_metrics )
      loss_exp_ce      = Lambda(lambda x: x[9], name='loss_exp_ce')( model_metrics ) 
      loss_kl          = Lambda(lambda x: x[10], name='loss_kl')( model_metrics )
      loss_akl         = Lambda(lambda x: x[11], name='loss_akl')( model_metrics )

    # Model Inputs
    inputs = [model_body.input, boxes_input, detectors_mask_input, matching_boxes_input]

    # Model Outputs
    loss_breakdown  = [total_loss, loss_conf, loss_class, loss_edl, loss_coord]
    edl_metrics     = [accrcy, ev_succ, ev_fail]
    edl_loss_components = [annealing_coeff, loss_exp_ce, loss_kl, loss_akl]
    outputs = loss_breakdown + edl_metrics + edl_loss_components

    # Build Model
    model = Model( inputs, outputs )

    # Custom metrics are outputs of Lamda layers so must hack into a metric
    # format as below... must be a better way to do this.

    # CUSTOM METRICS - EDL Metrics
    def class_acc(y_true, y_pred):
      return accrcy
    def succ_ev(y_true, y_pred):
      return ev_succ
    def fail_ev(y_true, y_pred):
      return ev_fail

    edl_metrics = [class_acc, succ_ev, fail_ev]

    # CUSTOM METRICS - TOTAL LOSS BREAKDOWN
    def conf_loss(y_true, y_pred):
      return loss_conf
    def class_loss(y_true, y_pred):
      return loss_class
    def edl_loss(y_true, y_pred):
      return loss_edl
    def coord_loss(y_true, y_pred):
      return loss_coord

    track_losses = [conf_loss, class_loss, edl_loss, coord_loss]


    # CUSTOM METRICS - EDL LOSS BREAKDOWN
    def coeff_anneal(y_true, y_pred):
      return annealing_coeff
    def exp_ce_loss(y_true, y_pred):
      return loss_exp_ce
    def kl_loss(y_true, y_pred):
      return loss_kl
    def akl_loss(y_true, y_pred):
      return loss_akl

    edl_loss_comps = [coeff_anneal, exp_ce_loss, kl_loss, akl_loss]

    metrics = edl_metrics + track_losses + edl_loss_comps

    return model_body, model, global_step, metrics

class MyCallback(Callback):
	def __init__(self, global_step):
		self.global_step = global_step

	def on_batch_end(self, batch, logs={}):
		K.set_value(self.global_step,  K.get_value(self.global_step) + 1)    

if __name__ == "__main__":
	voc07path = 'data/pascal_voc_07.hdf5'
	voc12path = 'data/pascal_voc_12.hdf5'

	# 2007 - 5011 images in total + 4952 test
	voc07 = h5py.File(voc07path, 'r')

	ims_train_07 = [im for im in voc07['train/images']]
	bxs_train_07 = [bx for bx in voc07['train/boxes']]
	ims_val_07   = [im for im in voc07['val/images']]
	bxs_val_07   = [bx for bx in voc07['val/boxes']]

	ims_tst_07   = [im for im in voc07['test/images']]
	bxs_tst_07   = [bx for bx in voc07['test/boxes']]

	# 2012 - 11,540 images in total
	voc12 = h5py.File(voc12path, 'r')

	ims_train_12 = [im for im in voc12['train/images']]
	bxs_train_12 = [bx for bx in voc12['train/boxes']]
	ims_val_12   = [im for im in voc12['val/images']]
	bxs_val_12   = [bx for bx in voc12['val/boxes']]

	# Training Set
	training_ims = ims_train_07 + ims_val_07 + ims_train_12 + ims_val_12
	training_bxs = bxs_train_07 + bxs_val_07 + bxs_train_12 + bxs_val_12


	# Validation Set
	validation_ims = ims_tst_07
	validation_bxs = bxs_tst_07


	og_sizes_train = np.array( [orig_size( im ) for im in training_ims] )
	og_sizes_val   = np.array( [orig_size( im ) for im in validation_ims])

	all_training_bxs = process_boxes(training_bxs, og_sizes_train)
	all_validation_bxs = process_boxes(validation_bxs, og_sizes_val)

	train_detectors_mask, train_matching_true_boxes = get_detector_mask(all_training_bxs, anchors)
	val_detectors_mask, val_matching_true_boxes = get_detector_mask(all_validation_bxs, anchors)

	X_train = { 'ims': np.array( training_ims ), 
	            'bxs': all_training_bxs,
	            'detectors': train_detectors_mask,
	            'true_bxs':  train_matching_true_boxes }

	X_val   = { 'ims': np.array( validation_ims ), 
	            'bxs': all_validation_bxs,
	            'detectors': val_detectors_mask,
	            'true_bxs':  val_matching_true_boxes }

	data = {'train': X_train, 'val': X_val

	### TRAIN
	num_epochs = 10
	batch_size = 32

	steps_per_epoch_fit = np.ceil(len(data[ 'train' ][ 'ims' ]) / batch_size)
	steps_per_epoch_val = np.ceil(len(data[ 'val'   ][ 'ims' ]) / batch_size)

	fit_gen = batch_generator( 'train', batch_size, shuffle = True)
	val_gen = batch_generator( 'val' , batch_size)

	model_body, model, global_step, metrics = create_model(anchors, class_names, reset_weights=False)

	model.compile( optimizer='adam', 
               loss =  {'yolo_loss': lambda y_true, y_pred: y_pred}, 
               metrics = metrics)

	annealing_step = (steps_per_epoch_fit*num_epochs) / 4

	hist = model.fit_generator(generator = fit_gen,
                           steps_per_epoch = steps_per_epoch_fit,
                           validation_data = val_gen,
                           validation_steps = steps_per_epoch_val,
                           epochs = num_epochs,
                           callbacks = [MyCallback(global_step)])

	model.save_weights('/model_data/trained.h5')
	history_path = '/model_data/trained'
	
	with open(history_path, 'wb') as fl:
		pickle.dump(hist.history, fl)