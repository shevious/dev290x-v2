#!/usr/bin/env python
import os
import numpy as np

#from keras import backend as K
#from tensorflow.compat.v1.keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo_keras.utils import *
from yolo_keras.model import *

# Get the COCO classes on which the model was trained
classes_path = "yolo_keras/coco_classes.txt"
with open(classes_path) as f:
    class_names = f.readlines()
    class_names = [c.strip() for c in class_names] 
num_classes = len(class_names)

# Get the anchor box coordinates for the model
anchors_path = "yolo_keras/yolo_anchors.txt"
with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)

# Set the expected image size for the model
model_image_size = (416, 416)

# Create YOLO model
home = os.path.expanduser(".")
model_path = os.path.join(home, "yolo.h5")
yolo_model = load_model(model_path, compile=False)

# Generate output tensor targets for bounding box predictions
# Predictions for individual objects are based on a detection probability threshold of 0.3
# and an IoU threshold for non-max suppression of 0.45
#input_image_shape = K.placeholder(shape=(2, ))

#boxes, scores, classes = yolo_eval(yolo_model.output, anchors, len(class_names), input_image_shape,
#                                    score_threshold=0.3, iou_threshold=0.45)

print("YOLO model ready!")

def detect_objects(image):
    
    # normalize and reshape image data
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    # Predict classes and locations using Tensorflow session
    '''
    sess = K.get_session()

    out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
    '''
    yolo_out = yolo_model(image_data, training=False)
    out_boxes, out_scores, out_classes = yolo_eval(yolo_out,
        anchors, len(class_names), [image.size[1], image.size[0]],
        score_threshold=0.3, iou_threshold=0.45)

    return out_boxes, out_scores, out_classes

def show_objects(image, out_boxes, out_scores, out_classes):
    import random
    from PIL import Image
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    #%matplotlib inline 
    
    # Set up some display formatting
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Plot the image
    img = np.array(image)
    #plt.figure() #this causes empty image
    fig, ax = plt.subplots(1, figsize=(12,9))
    plt.imshow(img)

    # Set up padding for boxes
    img_size = model_image_size[0]
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Use a random color for each class
    unique_labels = np.unique(out_classes)
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)

    # process each instance of each class that was found
    for i, c in reversed(list(enumerate(out_classes))):

        # Get the class name
        predicted_class = class_names[c]
        # Get the box coordinate and probability score for this instance
        box = out_boxes[i]
        score = out_scores[i]

        # Format the label to be added to the image for this instance
        label = '{} {:.2f}'.format(predicted_class, score)

        # Get the box coordinates
        top, left, bottom, right = box
        y1 = max(0, np.floor(top + 0.5).astype('int32'))
        x1 = max(0, np.floor(left + 0.5).astype('int32'))
        y2 = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        x2 = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # Set the box dimensions
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        
        # Add a box with the color for this class
        color = bbox_colors[int(np.where(unique_labels == c)[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=label, color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
        
    plt.axis('off')
    plt.show()
    
print("Functions ready")

import os
from PIL import Image

test_dir = "../../data/object_detection"
for image_file in os.listdir(test_dir):
    
    # Load image
    img_path = os.path.join(test_dir, image_file)
    image = Image.open(img_path)
    
    # Resize image for model input
    image = letterbox_image(image, tuple(reversed(model_image_size)))

    # Detect objects in the image
    out_boxes, out_scores, out_classes = detect_objects(image)

    # How many objects did we detect?
    print('Found {} objects in {}'.format(len(out_boxes), image_file))

    # Display the image with bounding boxes
    show_objects(image, out_boxes, out_scores, out_classes)
