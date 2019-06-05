import PIL.Image as Image
import numpy as np
import tensorflow as tf

print('tensorflow tf version:', tf.__version__)
print('tf_detector.py, tf.test.is_gpu_available:', tf.test.is_gpu_available())

MIN_DIM = 600
MAX_DIM = 1024

# Number of decimal places to round to for confidence and bbox coordinates
CONF_DIGITS = 3
COORD_DIGITS = 4


class TFDetector:

    detection_graph = None
    
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

    @staticmethod
    def resize_image(image):
        # resize the images since the client side renders them as small images too
        height, width = MIN_DIM, MAX_DIM
        return image.resize((width, height))  # PIL is lazy, so image only loaded here, not in open_image()

    @staticmethod
    def convert_numpy_floats_coords(np_array):
        new = []
        for i in np_array:
            new.append(round(float(i), COORD_DIGITS))
        return new

    def _generate_detections_batch(self, images, sess, image_tensor, box_tensor, score_tensor, class_tensor):
        print('_generate_detections_batch')
        np_images = [np.asarray(image, np.uint8) for image in images]
        images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)
        print('images_stacked shape: ', images_stacked.shape)

        # performs inference
        (box_tensor, score_tensor, class_tensor) = sess.run(
            [box_tensor, score_tensor, class_tensor],
            feed_dict={image_tensor: images_stacked})

        print('box_tensor shape: ', box_tensor.shape)
        return box_tensor, score_tensor, class_tensor

    def generate_detections_batch(self, images, image_ids, batch_size, detection_threshold):
        """
        Args:
            images: resized images to be processed by the detector
            image_ids: IDs for the images
            batch_size: mini-bath size to use during inference
            detection_threshold: detection confidence above which to record the detection result

        Returns:
            detections: list of detection entries with fields
                'category': str, numerical class label
                'conf': float rounded to CONF_DIGITS decimal places, score/confidence of the detection
                'bbox': list of floats rounded to COORD_DIGITS decimal places, relative coordinates [y1, x1, y2, x2]
            image_ids_completed: list of image_ids that were successfully processed
            failed_images_during_detection: list of image_ids that failed to process
        """

        # number of images should be small - all are loaded at once and a copy of resized version exists at one point
        # 2000 images are okay on a NC6s_v3
        print('tf_detector.py: generate_detections_batch...')

        # group the images into batches; image_batches is a list of lists
        image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
        image_id_batches = [image_ids[i:i + batch_size] for i in range(0, len(images), batch_size)]

        detections = []
        image_ids_completed = []  # image_ids minus ones that failed to process in this function
        failed_images_during_detection = []

        # start the TF session to process all images
        with tf.Session(graph=self.detection_graph) as sess:
            # get the operators
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            box_tensor = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            score_tensor = self.detection_graph.get_tensor_by_name('detection_scores:0')
            class_tensor = self.detection_graph.get_tensor_by_name('detection_classes:0')

            for b, (image_batch, image_id_batch) in enumerate(zip(image_batches, image_id_batches)):
                try:
                    print('tf_detector.py, processing batch {} out of {}.'.format(b + 1, len(image_batches)))

                    b_box, b_score, b_class = self._generate_detections_batch(image_batch,
                                                                              sess, image_tensor,
                                                                              box_tensor, score_tensor, class_tensor)
                    for i, image_id in enumerate(image_id_batch):
                        # apply the confidence threshold
                        boxes, scores, classes = b_box[i], b_score[i], b_class[i]
                        detections_cur_image = []  # will be empty for an image with no confident detections
                        max_detection_conf = 0.0
                        for b, s, c in zip(boxes, scores, classes):
                            if s > detection_threshold:
                                detection_entry = {
                                    'category': str(int(c)),  # use string type for the numerical class label, not int
                                    'conf': round(float(s), CONF_DIGITS),  # cast to float for json serialization
                                    'bbox': TFDetector.convert_numpy_floats_coords(b)
                                }
                                detections_cur_image.append(detection_entry)
                                if s > max_detection_conf:
                                    max_detection_conf = s

                        detections.append({
                            'file': image_id,
                            'max_detection_conf': round(float(max_detection_conf), CONF_DIGITS),
                            'detections': detections_cur_image
                        })
                        image_ids_completed.append(image_id)
                except Exception as e:
                    failed_images_during_detection.extent(image_id_batch)
                    print('tf_detector.py, one batch of images failed, exception: {}'.format(str(e)))
                    continue

        return detections, image_ids_completed, failed_images_during_detection


