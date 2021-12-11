# Postprocessing tools


For documentation on using the Output Manager app to split the batch processing API's output file into more manageable bites, see [here](./CameraTrapJsonManagerApp.md). 


The rest of thisi folder collects scripts to further process the output JSON file of the batch processing API.

-  `load_api_results.py` loads the output file into a Pandas dataframe and have functions to group entries by `seq_id` if ground truth JSON is provided as an instance of `IndexedJsonDb` (`data_management/cct_json_utils.py`) from a CCT format file. 

- `postprocess_batch_results.py` takes in the output file and renders an HTML page previewing the detection results. If ground truth is not provided, the sampled images are divided into `detections` and `non-detection` folders; if ground truth (in CCT format) is provided, results are divided into true/false positives/negatives, and a precision-recall curve is plotted. 

- `convert_output_format.py`: no longer used because the CSV output format is deprecated.

- `combine_api_outputs.py` merges two or more output JSON files into one. This is useful if a big batch had to be submitted in smaller batches, or if errored images were processed again successfully. 

- `separate_detections_into_folders.py` copies image files on your computer into animal/person/vehicle/empty/multiple folders according to confidence thresholds you specify for each category.

- `subset_json_detector_output.py` is primarily useful for [Timelapse](../integration/timelapse.md) users. 

    - When you load detection output in Timelapse, you often want to work on only a subset of images in a certain folder, not the full set of images referred to in the output file. This allows Timelapse to construct its database more quickly.

    - This script splits the output file into smaller ones each containing only results for images in a folder. You can specify the folder level to work at, as well as making the path to the images relative, or replace a part of the path to match how they are stored on your current computer.

