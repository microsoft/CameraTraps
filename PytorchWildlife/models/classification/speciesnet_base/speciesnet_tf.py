import multiprocessing as mp

from speciesnet import SpeciesNet
from speciesnet.utils import prepare_instances_dict

from operator import itemgetter
from ..base_classifier import BaseClassifierInference 

__all__ = ["SpeciesNetTFInference"]

def get_by_key(lst, key, value):
    return next(filter(lambda x: itemgetter(key)(x) == value, lst), None)


class SpeciesNetTFInference(BaseClassifierInference):
    """
    Inference module for the PlainResNet Classifier.
    """
    def __init__(self, version='v4.0.0a', run_mode='multi_thread', geofence=True):
        super(SpeciesNetTFInference, self).__init__()

        self.model_url = 'kaggle:google/speciesnet/keras/{}'.format(version)
        self.run_mode = run_mode 
        self.geofence = geofence

        self.progress_bars = True

        try:
            mp.set_start_method('spawn')
        except RuntimeError as e:
            if "context has already been set" in str(e):
                pass  # Context is already set, so skip silently
            else:
                raise

        self.model = SpeciesNet(
            self.model_url,
            components='classifier',
            geofence=self.geofence,
            # target_species_txt=target_species_txt,
            multiprocessing=(self.run_mode == "multi_process"),
        )

    def detections_dict_generation(self, det_results):
        detections_dict = {}

        for det in det_results:
            det['filepath'] = det['img_id']
            det['detections'] = [{'bbox' : [b[0], b[1], b[2] - b[0], b[3] - b[1],]}
                                 for b in det['normalized_coords']]
            detections_dict[det['filepath']] = det 

        return detections_dict

    def results_generation(self, predictions_dict, det_results):
        clf_results = []
        for pred in predictions_dict['predictions']:
            det = get_by_key(det_results, 'img_id', pred['filepath']) 
            for _ in range(len(det['normalized_coords'])):
                clf_results.append({
                    'img_id': pred['filepath'],
                    'prediction': pred['classifications']['classes'][0].split(';')[-1],
                    'confidence': pred['classifications']['scores'][0]
                })
        return clf_results

    def single_image_classification(self, file_path, det_results=None):

        instances_dict = prepare_instances_dict(
            filepaths=[file_path],
        )

        predictions_dict = self.model.classify(
            instances_dict=instances_dict,
            detections_dict=self.detections_dict_generation([det_results]) if det_results else None,
            run_mode=self.run_mode,
            batch_size=1,
            progress_bars=self.progress_bars,
        )
        return self.results_generation(predictions_dict, [det_results])

    def batch_image_classification(self, data_path, batch_size=8, det_results=None):

        instances_dict = prepare_instances_dict(
            folders=[data_path]
        )

        predictions_dict = self.model.classify(
            instances_dict=instances_dict,
            detections_dict=self.detections_dict_generation(det_results) if det_results else None,
            run_mode=self.run_mode,
            batch_size=batch_size,
            progress_bars=self.progress_bars,
        )

        return self.results_generation(predictions_dict, det_results)