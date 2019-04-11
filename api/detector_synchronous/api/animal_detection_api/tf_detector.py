import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from animal_detection_api import api_config

tf.logging.set_verbosity(tf.logging.ERROR)


class TFDetector:

    def __init__(self, checkpoint):
        self.detection_graph = self.load_model(checkpoint)
        self.session = tf.Session(graph=self.detection_graph)

    def load_model(self, checkpoint):
        """Loads a detection model (i.e., create a graph) from a .pb file.

        Args:
            checkpoint: .pb file of the model.

        Returns: the loaded graph.

        """
        print('tf_detector.py: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('tf_detector.py: Detection graph loaded.')

        return detection_graph

    @staticmethod
    def open_image(input):
        """Opens an image in binary format using PIL.Image and convert to RGB mode.

        Args:
            input: an image in binary format read from the POST request's body or
                path to an image file (anything that PIL can open)

        Returns:
            an PIL image object in RGB mode
        """
        image = Image.open(input)
        if image.mode not in ('RGBA', 'RGB'):
            raise AttributeError('Input image not in RGBA or RGB mode and cannot be processed.')
        if image.mode == 'RGBA':
            # PIL.Image.convert() returns a converted copy of this image
            image = image.convert(mode='RGB')
        return image

    def _generate_detections_batch(self, images, image_tensor, box, score):
        print('_generate_detections_batch')
        np_images = [np.asarray(image, np.uint8) for image in images]
        images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)
        print('images_stacked shape: ', images_stacked.shape)

        # performs inference
        (box, score) = self.session.run(
            [box, score],
            feed_dict={image_tensor: images_stacked})

        print('box shape: ', box.shape)
        print('score shape: ', score.shape)
        return box, score

    def generate_detections_batch(self, images):
        # number of images should be small - all are loaded at once and a copy of resized version exists at one point
        print('tf_detector.py: generate_detections_batch...')

        # resize the images since the client side renders them as small images too
        height, width = api_config.MIN_DIM, api_config.MAX_DIM
        resized_images = []
        for image in images:
            resized_images.append(image.resize((width, height)))  # first time image is loaded by PIL
        images = resized_images

        # group the images into batches; image_batches is a list of lists
        batch_size = api_config.GPU_BATCH_SIZE
        image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
        print('Length of image_batches: ', len(image_batches))
        detections = []


        # get the operators to go in the fetch list
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        box = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        score = self.detection_graph.get_tensor_by_name('detection_scores:0')

        for image_batch in image_batches:
            b_box, b_score = self._generate_detections_batch(image_batch, image_tensor, box, score)
            for i, image in enumerate(image_batch):
                detections.append({
                    'box': b_box[i],
                    'score': b_score[i],
                    'image': image  # save the opened image for rendering
                })

        print('tf_detector.py: generate_detections_batch finished')
        return detections

    @staticmethod
    def render_bounding_boxes(boxes, scores, image, label_map={}, confidence_threshold=0.5):
        """Renders bounding boxes, label and confidence on an image if confidence is above the threshold.

        Args:
            boxes, scores, classes:  outputs of generate_detections.
            image: PIL.Image object, output of generate_detections.
            label_map: optional, mapping the numerical label to a string name.
            confidence_threshold: threshold above which the bounding box is rendered.

        image is modified in place!
        """
        display_boxes = []
        display_strs = []  # list of list, one list of strings for each bounding box (to accommodate multiple labels)
        for box, score in zip(boxes, scores):
            if score > confidence_threshold:
                # print('To draw score {}, box {}'.format(score, box))
                display_boxes.append(box)
                clss = 1  # only 1 class "animal"
                label = label_map[clss] if clss in label_map else str(clss)
                displayed_label = '{}: {}%'.format(label, round(100*score))
                display_strs.append([displayed_label])

        display_boxes = np.array(display_boxes)
        TFDetector.draw_bounding_boxes_on_image(image, display_boxes, display_str_list_list=display_strs)

    # the following two functions are from https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

    @staticmethod
    def draw_bounding_boxes_on_image(image,
                                     boxes,
                                     color='red',
                                     thickness=4,
                                     display_str_list_list=()):
      """Draws bounding boxes on image.

      Args:
        image: a PIL.Image object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
               The coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings.
                               a list of strings for each bounding box.
                               The reason to pass a list of strings for a
                               bounding box is that it might contain
                               multiple labels.

      Raises:
        ValueError: if boxes is not a [N, 4] array
      """
      boxes_shape = boxes.shape
      if not boxes_shape:
        return
      if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        #print('Input must be of size [N, 4], but is ' + str(boxes_shape))
        return  # no object detection on this image, return
      for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
          display_str_list = display_str_list_list[i]
          TFDetector.draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                   boxes[i, 3], color, thickness, display_str_list)

    @staticmethod
    def draw_bounding_box_on_image(image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color='red',
                                   thickness=4,
                                   display_str_list=(),
                                   use_normalized_coordinates=True):
      """Adds a bounding box to an image.

      Bounding box coordinates can be specified in either absolute (pixel) or
      normalized coordinates by setting the use_normalized_coordinates argument.

      Each string in display_str_list is displayed on a separate line above the
      bounding box in black text on a rectangle filled with the input 'color'.
      If the top of the bounding box extends to the edge of the image, the strings
      are displayed below the bounding box.

      Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                          (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
          ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
          coordinates as absolute.
      """
      draw = ImageDraw.Draw(image)
      im_width, im_height = image.size
      if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
      else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
      draw.line([(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)], width=thickness, fill=color)

      try:
        font = ImageFont.truetype('arial.ttf', 24)
      except IOError:
        font = ImageFont.load_default()

      # If the total height of the display strings added to the top of the bounding
      # box exceeds the top of the image, stack the strings below the bounding box
      # instead of above.
      display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
      # Each display_str has a top and bottom margin of 0.05x.
      total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

      if top > total_display_str_height:
        text_bottom = top
      else:
        text_bottom = bottom + total_display_str_height
      # Reverse list and print from bottom to top.
      for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin
