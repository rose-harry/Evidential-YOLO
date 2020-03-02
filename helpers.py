def resize(im):
  with PIL.Image.open(io.BytesIO(im)) as image:
    resized = image.resize((416, 416), PIL.Image.BICUBIC)
    resized = np.array( resized )
  return resized 

def orig_size(im):
  with PIL.Image.open(io.BytesIO(im)) as image:
    dims = [image.width, image.height]
  return dims

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def process_boxes(boxes, orig_image_sizes):

    # Box preprocessing.
    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
    boxes = [box.reshape((-1, 5)) for box in boxes]
    # Get extents as y_min, x_min, y_max, x_max, class for comparision with
    # model output.
    boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

    # Get box parameters as x_center, y_center, box_width, box_height, class.
    boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
    boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
    boxes_xy = [boxxy / orig_image_sizes[i] for i, boxxy in enumerate(boxes_xy)]
    boxes_wh = [boxwh / orig_image_sizes[i] for i, boxwh in enumerate(boxes_wh)]
    boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

    # find the max number of boxes
    max_boxes = 0
    for boxz in boxes:
        if boxz.shape[0] > max_boxes:
            max_boxes = boxz.shape[0]

    # add zero pad for training
    for i, boxz in enumerate(boxes):
        if boxz.shape[0]  < max_boxes:
            zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
            boxes[i] = np.vstack((boxz, zero_padding))

    return np.array(boxes)
  
def shuffle_data( sect ):
  assert data[ sect ]['ims'].shape[0] == data[ sect ]['bxs'].shape[0] == \
    data[ sect ]['detectors'].shape[0] == data[ sect ]['true_bxs'].shape[0]
  
  p = np.random.permutation( data[ sect ]['ims'].shape[0] )
  
  # Permute
  for key in data[ sect ].keys():
    data[ sect ][ key ] = data[ sect ][ key ][p]
    
def batch_generator( sect, batch_size, shuffle=False):
  '''
  Given size of training set, need to load in batches.
  '''
  
  # Shuffle on train
  if shuffle:
    shuffle_data(sect)
  
  number_of_batches = np.ceil( len(data[ sect ][ 'ims' ]) / batch_size)
  counter = 0

  while True:

      idx_start = batch_size * counter
      idx_end = batch_size * (counter + 1)
      x0 = np.array( [ resize(im) for im in data[ sect ][ 'ims' ][ idx_start:idx_end ] ]  ) / 255.


      x1 = data[ sect ][ 'bxs' ][ idx_start:idx_end ]
      x2 = data[ sect ][ 'detectors' ][ idx_start:idx_end ]
      x3 = data[ sect ][ 'true_bxs' ][ idx_start:idx_end ]

      x_batch = [x0, x1, x2, x3]
      y_batch = np.zeros( len(x0) )

      counter += 1

      yield x_batch, y_batch

      if counter == number_of_batches:
        counter = 0  