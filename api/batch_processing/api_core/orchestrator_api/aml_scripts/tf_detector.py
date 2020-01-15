import PIL.Image as Image
import numpy as np
import tensorflow as tf

print('tensorflow tf version:', tf.__version__)
print('tf_detector.py, tf.test.is_gpu_available:', tf.test.is_gpu_available())


batch_size = 1

# Number of decimal places to round to for confidence and bbox coordinates
CONF_DIGITS = 3
COORD_DIGITS = 4


class TFDetector:

    def __init__(self, model_path):
        self.detection_graph = self.load_model(model_path)

    def load_model(self, model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.

        Args:
            model_path: .pb file of the model.

        Returns: the loaded graph.

        """
        print('tf_detector.py: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('tf_detector.py: Detection graph loaded.')

        return detection_graph

    @staticmethod
    def open_image(input_file):
        """Opens an image in binary format using PIL.Image and convert to RGB mode.

        Args:
            input_file: an image in binary format read from the POST request's body or
                path to an image file (anything that PIL can open)

        Returns:
            an PIL image object in RGB mode
        """
        image = Image.open(input_file)
        if image.mode not in ('RGBA', 'RGB'):
            raise AttributeError('Input image not in RGBA or RGB mode and cannot be processed.')
        if image.mode == 'RGBA':
            # PIL.Image.convert() returns a converted copy of this image
            image = image.convert(mode='RGB')
        return image

    @staticmethod
    def round_and_make_float(d):
        return round(float(d), COORD_DIGITS)

    @staticmethod
    def convert_coords(np_array):
        """ Two functionalities: convert the numpy floats to Python floats, and also change the coordinates from
        [y1, x1, y2, x2] to [x1, y1, width_box, height_box] (in relative coordinates still).

        Args:
            np_array: array of predicted bounding box coordinates from the TF detector

        Returns: array of predicted bounding box coordinates as Python floats and in [x1, y1, width_box, height_box]

        """
        # change from [y1, x1, y2, x2] to [x1, y1, width_box, height_box]
        width_box = np_array[3] - np_array[1]
        height_box = np_array[2] - np_array[0]

        new = [np_array[1], np_array[0], width_box, height_box]  # cannot be a numpy array; needs to be a list

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d)
        return new

    def _generate_detections_batch(self, images, sess, image_tensor, box_tensor, score_tensor, class_tensor):
        print('_generate_detections_batch')
        np_images = [np.asarray(image, np.uint8) for image in images]
        images_stacked = np.expand_dims(np_images[0], axis=0)

        print('images_stacked shape: ', images_stacked.shape)

        # performs inference
        (box_tensor, score_tensor, class_tensor) = sess.run(
            [box_tensor, score_tensor, class_tensor],
            feed_dict={image_tensor: images_stacked})

        print('box_tensor shape: ', box_tensor.shape)
        return box_tensor, score_tensor, class_tensor

    def generate_detections_batch(self, images, image_ids, detection_threshold,
                                  image_metas=None, metadata_available=False):
        """
        Args:
            images: resized images to be processed by the detector
            image_ids: list of strings, IDs for the images
            detection_threshold: detection confidence above which to record the detection result
            image_metas: list of strings, same length as image_ids
            metadata_available: is image_metas actually available (if not, image_metas can be a list of None)

        Returns:
            detections: list of detection entries with fields
                'category': str, numerical class label
                'conf': float rounded to CONF_DIGITS decimal places, score/confidence of the detection
                'bbox': list of floats rounded to COORD_DIGITS decimal places, relative coordinates
                        [x, y, width_box, height_box]
            image_ids_completed: list of image_ids that were successfully processed
            failed_images: list of image_ids for images that failed to process
            failed_metas: list of image_metas for images that failed to process
        """

        # number of images should be small - all are loaded at once and a copy of resized version exists at one point
        # 2000 images are okay on a NC6s_v3
        print('tf_detector.py: generate_detections_batch...')

        # group the images into batches; image_batches is a list of lists
        image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

        # we keep track of the image_ids (and image_metas when available) to be able to output the list of failed images
        image_id_batches = [image_ids[i:i + batch_size] for i in range(0, len(images), batch_size)]
        if image_metas is None:
            image_metas = [None] * len(images)
        image_meta_batches = [image_metas[i:i + batch_size] for i in range(0, len(images), batch_size)]

        detections = []
        failed_images = []
        failed_metas = []

        # start the TF session to process all images
        with tf.Session(graph=self.detection_graph) as sess:
            # get the operators
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            box_tensor = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            score_tensor = self.detection_graph.get_tensor_by_name('detection_scores:0')
            class_tensor = self.detection_graph.get_tensor_by_name('detection_classes:0')

            for i_batch, (image_batch, image_id_batch, image_meta_batch) in enumerate(zip(
                    image_batches, image_id_batches, image_meta_batches)):
                try:
                    print('tf_detector.py, processing batch {} out of {}.'.format(i_batch + 1, len(image_batches)))

                    b_box, b_score, b_class = self._generate_detections_batch(image_batch,
                                                                              sess, image_tensor,
                                                                              box_tensor, score_tensor, class_tensor)
                    for i, (image_id, image_meta) in enumerate(zip(image_id_batch, image_meta_batch)):
                        # apply the confidence threshold
                        boxes, scores, classes = b_box[i], b_score[i], b_class[i]
                        detections_cur_image = []  # will be empty for an image with no confident detections
                        max_detection_conf = 0.0
                        for b, s, c in zip(boxes, scores, classes):
                            if s > detection_threshold:
                                detection_entry = {
                                    'category': str(int(c)),  # use string type for the numerical class label, not int
                                    'conf': round(float(s), CONF_DIGITS),  # cast to float for json serialization
                                    'bbox': TFDetector.convert_coords(b)
                                }
                                detections_cur_image.append(detection_entry)
                                if s > max_detection_conf:
                                    max_detection_conf = s

                        detection = {
                            'file': image_id,
                            'max_detection_conf': round(float(max_detection_conf), CONF_DIGITS),
                            'detections': detections_cur_image
                        }
                        if metadata_available:
                            detection['meta'] = image_meta
                        detections.append(detection)

                except Exception as e:
                    failed_images.extend(image_id_batch)
                    failed_metas.extend(image_meta_batch)
                    print('tf_detector.py, one batch of images failed, exception: {}'.format(str(e)))
                    continue

        return detections, failed_images, failed_metas
