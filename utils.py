import tensorflow as tf
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt


def display(img):
    dpi = mpl.rcParams['figure.dpi']
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    dims = img.shape
    h, w = dims[0], dims[1]
    figsize = w / float(dpi), h / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap='gray')
    plt.show()

def _draw_box(img, box, label=None, box_color=None, 
                             label_color=None, thickness=2):
    box = box.astype(np.int32)
    x1, y1, x2, y2 = box
    img_height = img.shape[0]
    if box_color is None:
        box_color = [np.random.randint(0,256) for _ in range(3)]

    img = cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

    if label is not None:
        if label_color is None:
            label_color = (128,128,128)
        img = cv2.putText(img, label, (x1, y1-12), 0, 1e-3 * img_height, label_color, thickness)
    return img


def draw_boxes(img, boxes, labels=None, box_color=None,
                             label_color=None, thickness=2):
    # img = np.ascontiguousarray(img)
    boxes = np.atleast_2d(boxes)
    if labels is not None:
        if len(boxes) != len(labels):
            raise Exception("No of boxes must be equal to No of Labels")
    for i, box in enumerate(boxes):
        label = None if labels == None else labels[i]
        img = _draw_box(img, box, label, box_color, label_color, thickness)
    return img


def calculate_iou(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = tf.math.maximum(box1_x1, box2_x1)
    y1 = tf.math.maximum(box1_y1, box2_y1)
    x2 = tf.math.minimum(box1_x2, box2_x2)
    y2 = tf.math.minimum(box1_y2, box2_y2)

    intersection = tf.clip_by_value(x2-x1, 0.0, float('inf')) * \
                   tf.clip_by_value(y2-y1, 0.0, float('inf'))

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def image_hw_when_resized(original_h, original_w, new_h, new_w):
    scale = min(new_w/original_w, new_h/original_h)
    new_w = int(original_w*scale)
    new_h = int(original_h*scale)
    return new_h, new_w

def resize_image(image, new_h, new_w):
    return cv2.resize(image, (new_w,new_h))


def right_bottom_padding(image, new_h, new_w):
    h, w = image.shape[:2]
    if h > new_h or w > new_w:
        raise ValueError("image dims before padding must be <= image dims after padding")
    padded_image = np.full((new_h, new_w, 3), 128, dtype=np.uint8)
    padded_image[0:h, 0:w, :] = image
    return padded_image

    
def letterbox_padding(image, new_h, new_w):
    h, w = image.shape[:2]
    if h > new_h or w > new_w:
        raise ValueError("image dims before padding must be <= image dims after padding")
    x = (new_w - w)//2
    y = (new_h - h)//2
    padded_image = np.full((new_h, new_w, 3), 128, dtype=np.uint8)
    padded_image[y:y+h, x:x+w, :] = image
    return padded_image

def create_letterbox_image(image, new_h, new_w):
    h, w = image.shape[:2]
    temp_h, temp_w = image_hw_when_resized(h, w, new_h, new_w)
    image = resize_image(image, temp_h, temp_w)
    return letterbox_padding(image, new_h, new_w)

def correct_boxes_when_letterpadded(boxes, image_h, image_w, net_h, net_w):
    ''' image_h, image_w are dimensions of original image
        net_h, net_w are dimensions of image that network accepts Ex: (416, 416)
        new_h, new_w are dimensions of image after it's resized by keeping aspect_ration intact'''
    
    new_h, new_w = image_hw_when_resized(image_h, image_w, net_h, net_w)
    org_scale = np.array([image_w, image_h, image_w, image_h], dtype=np.float32)
    new_scale = np.array([new_w, new_h, new_w, new_h], dtype=np.float32)
    net_scale = np.array([net_w, net_h, net_w, net_h], dtype=np.float32)

    offset_x = (net_w-new_w)//2.0
    offset_y = (net_h-new_h)//2.0
    offsets = np.array([offset_x, offset_y, offset_x, offset_y])
    
    #check if boxes provided are normalized
    is_normalized = np.all([boxes >= 0.0, boxes <=1.0])
    
    if is_normalized:
        new_boxes = (boxes*new_scale + offsets)
    else:
        new_boxes = (boxes*np.reciprocal(org_scale)*new_scale + offsets)
    scaled_new_boxes = new_boxes * np.reciprocal(net_scale)
    return scaled_new_boxes
