# Camera trap batch processing API user guide

We offer a service for processing a large quantity of camera trap images using our [MegaDetector](https://github.com/Microsoft/CameraTraps#megadetector) by calling an API, documented here. The output is most helpful for separating empty from non-empty images based on some detector confidence threshold that you select, and putting bounding boxes around animals so that manual review can proceed faster.

You can process a batch of up to two million images in one request to the API. If in addition you have some images that are labeled, we can evaluate the performance of the MegaDetector on your labeled images (see [Post-processing tools](#post-processing-tools)).

If you would like to try automating species identification as well, we can train a project-specific classifier based on the output of this detector API and the labels you provide. The species classification predictions will be added to the detector API output json file.

All references to &ldquo;container&rdquo; in this document refer to [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/) containers. 


## Processing time

It takes about 0.8 seconds per image per machine, and we have at most 16 machines that can process them in parallel. So if no one else is using the service and you&rsquo;d like to process 1 million images, it will take 1,000,000 * 0.8 / (16 * 60 * 60) = 14 hours. 


## API

### API endpoints

The endpoints of this API are available at

```
http://URL/v3/camera-trap/detection-batch
```

#### `/request_detections`
To submit a request for batch processing, make a POST call to

```http://URL/v3/camera-trap/detection-batch/request_detections```.

with a json body containing input fields defined below. The API will return with a json response very quickly to give you a RequestID (UUID4) representing the request you have submitted, for example:
```json
{
  "request_id": "ea26326e-7e0d-4524-a9ea-f57a5799d4ba"
}
```
or an error message, if your inputs are not acceptable:
```json
{
  "error": "error message."
}
```


#### `/task`
Check the status of your request by calling the `/task` endpoint via a GET call, passing in your RequestID:

```http://URL/v3/camera-trap/detection-batch/task/RequestID```

This returns a json with the fields `Status`, `TaskId` (which is the `request_id` in this document), and a few others. The `Status` field is a json object with the following fields: 

- `request_status`: one of `running`, `failed`, `problem`, and `completed`. 

    - The status `failed` indicates that the images have not been submitted to the cluster for processing, and so you can go ahead and call the endpoint again, correcting your inputs according to the error message returned with the status. 
    - The status `problem` indicates that the images have already been submitted for processing but the API encountered an error while monitoring progress; in this case, *please do not retry*; contact us to retrieve your results so that no unnecessary processing would occupy the cluster (`message` field will mention "please contact us").

- `message`: a longer string describing the `request_status` and any errors; when the request is completed, the URLs to the output files will also be here (see [Outputs](#23-outputs) section below).

- `time`: timestamp in UTC.


#### `/supported_model_versions`
Check which versions of the MegaDetector are supported by this API by making a GET call to 

```http://URL/v3/camera-trap/detection-batch/supported_model_versions```


#### `/default_model_version`
Check which versions of the MegaDetector is used by default by making a GET call to

```http://URL/v3/camera-trap/detection-batch/default_model_version```


#### Canceling a request
Not yet supported. If you have accidentally submitted a request, please contact us to cancel it.


### API inputs

| Parameter                | Is required | Type | Explanation                 |
|--------------------------|-------------|-------|----------------------------|
| input_container_sas      | Yes<sup>1</sup>         | string | SAS URL with list and read permissions to the Blob Storage container where the images are stored. |
| images_requested_json_sas | No<sup>1</sup>        | string | SAS URL with list and read permissions to a json file in Blob Storage. See below for explanation of the content of the json to provide. |
| image_path_prefix        | No          | string | Only process images whose full path starts with `image_path_prefix` (case-_sensitive_). Note that any image paths specified in `images_requested_json_sas` will need to be the full path from the root of the container, regardless whether `image_path_prefix` is provided. |
| first_n                  | No          | int | Only process the first `first_n` images. Order of images is not guaranteed, but is likely to be alphabetical. Set this to a small number to avoid taking time to fully list all images in the blob (about 15 minutes for 1 million images) if you just want to try this API. |
| sample_n                | No          | int | Randomly select `sample_n` images to process. |
| model_version           | No          | string | Version of the MegaDetector model to use. Default is the most updated stable version (check using the `/default_model_version` endpoint). Supported versions are available at the `/supported_model_versions` endpoint.|
| request_name            | No          | string | A string (letters, digits, `_`, `-` allowed, max length 92 characters) that will be appended to the output file names to help you identify the resulting files. A timestamp in UTC (`%Y%m%d%H%M%S`) of the time of submission will be appended to the resulting files automatically. |
| use_url                  | No         | bool | Please set to `true` if you are providing public image URLs. |
| caller                  | Yes         | string | An identifier that we use to whitelist users for now. |

<sup>1</sup> There are two ways of giving the API access to your images. 

1 - If you have all your images in a container in Azure Blob Storage, provide the parameter `input_container_sas` as described above. This means that your images do not have to be at publicly accessible URLs. In this case, the json pointed to by `images_requested_json_sas` should look like:
```json
[
  "Season1/Location1/Camera1/image1.jpg", 
  "Season1/Location1/Camera1/image2.jpg"
]
```
Only images whose paths are listed here will be processed.

2 - If your images are stored elsewhere and you can provide a publicly accessible URL to each of them, you do not need to specify `input_container_sas`. Instead, list the URLs to all the images (instead of their paths) you&rsquo;d like to process in the json at `images_requested_json_sas`.


#### Storing metadata

We can store a (short) string of metadata with each image path or URL. The json at `images_requested_json_sas` should then look like:
```json
[
  ["Season1/Location1/Camera1/image1.jpg", "metadata_string1"], 
  ["Season1/Location1/Camera1/image2.jpg", "metadata_string2"]
]
``` 


#### Other notes and example  

- Only images with file name ending in &lsquo;.jpg&rsquo; (case insensitive) will be processed, so please make sure the file names are compliant before you upload them to the container (you cannot rename a blob without copying it entirely once it is in Blob Storage). 

- By default we process all such images in the specified container. You can choose to only process a subset of them by specifying the other input parameters, and the images will be filtered out accordingly in this order:
    - `images_requested_json_sas`
    - `image_path_prefix`
    - `first_n`
    - `sample_n`
    
    - For example, if you specified both `images_requested_json_sas` and `first_n`, only images that are in your provided list at `images_requested_json_sas` will be considered, and then we process the `first_n` of those.

Example body of the POST request:
```json
{
  "input_container_sas": "https://storageaccountname.blob.core.windows.net/container-name?se=2019-04-23T01%3A30%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=A_LONG_STRING",
  "images_requested_json_sas": "https://storageaccountname2.blob.core.windows.net/container-name2/possibly_in_a_folder/my_list_of_images.json?se=2019-04-19T20%3A31%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=ANOTHER_LONG_STRING",
  "image_path_prefix": "2020/Alberta",
  "first_n": 100000,
  "request_name": "Alberta_2020",
  "model_version": "4.1",
  "caller": "allowlisted_user_x"
}
```

You can manually call the API using applications such as Postman:

![Screenshot of Azure Storage Explorer used for generating SAS tokens with read and list permissions](./images/Postman_screenshot.png)


#### How to obtain a SAS token

You can easily generate a [SAS token](https://docs.microsoft.com/en-us/azure/storage/common/storage-dotnet-shared-access-signature-part-1) to a container or a particular blob (a file in blob storage) using the desktop app [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/) (available on Windows, macOS and Linux). You can also issue SAS tokens programmatically by using the [Azure Storage SDK for Python](https://azure-storage.readthedocs.io/ref/azure.storage.blob.baseblobservice.html#azure.storage.blob.baseblobservice.BaseBlobService.generate_blob_shared_access_signature).


Using Storage Explorer, right click on the container or blob you&rsquo;d like to grant access for, and choose &ldquo;Get Shared Access Signature...&rdquo;. On the dialog window that appears, 
- cross out the &ldquo;Start time&rdquo; field if you will be using the SAS token right away
- set the &ldquo;Expiry time&rdquo; to a date in the future, about a month ahead is reasonable. The SAS token needs to be valid for the duration of the batch processing request. 
- make sure &ldquo;Read&rdquo; and &ldquo;List&rdquo; are checked under &ldquo;Permissions&rdquo; (see screenshot) 

Click &ldquo;Create&rdquo;, and the &ldquo;URL&rdquo; field on the next screen is the value required for `input_container_sas` or `images_requested_json_sas`. 

![Screenshot of Azure Storage Explorer used for generating SAS tokens with read and list permissions](./images/SAS_screenshot.png)


### API outputs

Once your request is submitted and parameters validated, the API divides all images into shards of about 2000 images each, and send them to an Azure Machine Learning Service compute cluster for processing. Another process will monitor how many shards have been evaluated, checking every 15 minutes, and update the status of the request, which you can check via the `/task` endpoint. 

When all shards have finished processing, the `status` returned by the `/task` endpoint will have the `request_status` field as `completed`, and the `message` field contain a string that can be loaded as a json with 3 fields, each containing an URL to a downloadable file. The returned body looks like

```json
{
    "Status": {
        "request_status": "completed",
        "message": {
            "num_failed_shards": 0,
            "output_file_urls": {
                "detections": "https://cameratrap.blob.core.windows.net/async-api-internal/ee26326e-7e0d-4524-a9ea-f57a5799d4ba/ee26326e-7e0d-4524-a9ea-f57a5799d4ba_detections_4_1_on_test_images_20200709211752.json?sv=2019-02-02&sr=b&sig=key1",
                "failed_images": "https://cameratrap.blob.core.windows.net/async-api-internal/ee26326e-7e0d-4524-a9ea-f57a5799d4ba/ee26326e-7e0d-4524-a9ea-f57a5799d4ba_failed_images_4_1_on_test_images_20200709211752.json?sv=2019-02-02&sr=b&sig=key2",
                "images": "https://cameratrap.blob.core.windows.net/async-api-internal/ee26326e-7e0d-4524-a9ea-f57a5799d4ba/ee26326e-7e0d-4524-a9ea-f57a5799d4ba_images.json?sv=2019-02-02&sr=b&sig=key3"
            }
        },
        "time": "2020-07-09 21:27:17"
    },
    "Timestamp": "2020-07-09 21:27:17",
    "Endpoint": "/v3/camera-trap/detection-batch/request_detections",
    "TaskId": "ea26326e-7e0d-4524-a9ea-f57a5799d4ba"
}
```
 
 You can parse it to obtain the URLs:
```python
task_status = body['Status']
assert task_status['request_status'] == 'completed'
message = task_status['message']
assert message['num_failed_shards'] == 0

output_files = message['output_file_urls']
url_to_result = output_files['detections']
url_to_failed_images = output_files['failed_images']
url_to_all_images_processed = output_files['images']

```
Note that the field `Status` in the returned body is capitalized (since July 2020).

These URLs are valid for 90 days from the time the request has finished. If you neglected to retrieve them before the links expired, contact us with the RequestID and we can send the results to you. Here are the 3 files to expect:


| File name                | Description | 
|--------------------------|-------------|
| requestID_detections_requestName_timestamp.json | Contains the predictions produced by the detector, in the format defined below.   |
| requestID_failed_images_requestName_timestamp.json | Contains a list of full paths to images in the blob that the API failed to open or process, possibly because they are corrupted. |
| requestID_images.json | Contains a list of the full paths to all images that the API was supposed to process, based on the content of the container at the time the API was called and the filtering parameters provided. |


#### Batch processing API output format

The output of the detector is saved in `requestID_detections_requestName_timestamp.json`. The `classifications` fields will be added if a classifier was trained for your project and applied to the images. Example output with both detection and classification results:

```json
{
    "info": {
        "detector": "megadetector_v3",
        "detection_completion_time": "2019-05-22 02:12:19",
        "classifier": "ecosystem1_v2",
        "classification_completion_time": "2019-05-26 01:52:08",
        "format_version": "1.0"
    },
    "detection_categories": {
        "1": "animal",
        "2": "person",
        "3": "vehicle"
    },
    "classification_categories": {
        "0": "fox",
        "1": "elk",
        "2": "wolf",
        "3": "bear",
        "4": "moose"
    },
    "images": [
        {
            "file": "path/from/base/dir/image1.jpg",
            "meta": "a string of metadata if it was available in the list at images_requested_json_sas",
            "max_detection_conf": 0.926,
            "detections": [
                {
                    "category": "1",
                    "conf": 0.926,
                    "bbox": [0.0, 0.2762, 0.1539, 0.2825], 
                    "classifications": [
                        ["3", 0.901],
                        ["1", 0.071],
                        ["4", 0.025]
                    ]
                },
                {
                    "category": "1",
                    "conf": 0.061,
                    "bbox": [0.0451, 0.1849, 0.3642, 0.4636]
                }
            ]
        },
        {
            "file": "/path/from/base/dir/image2.jpg",
            "meta": "",
            "max_detection_conf": 0,
            "detections": []
        }
    ]
}
```

A full output example computed on the Snapshot Serengeti data can be found [here](http://dolphinvm.westus2.cloudapp.azure.com/data/snapshot_serengeti/serengeti_val_detections_from_pkl_MDv1_20190528_w_classifications.json).


##### Detector outputs

The bounding box in the `bbox` field is represented as

```
[x_min, y_min, width_of_box, height_of_box]
```

where `(x_min, y_min)` is the upper-left corner of the detection bounding box, with the origin in the upper-left corner of the image. The coordinates and box width and height are *relative* to the width and height of the image. Note that this is different from the coordinate format used in the [COCO Camera Traps](data_management/README.md) databases, which are in absolute coordinates. 

The detection category `category` can be interpreted using the `detection_categories` dictionary. 

Detection categories not listed here are allowed by this format specification, but should be treated as "no detection".

When the detector model detects no animal (or person or vehicle), the confidence `conf` is shown as 0.0 (not confident that there is an object of interest) and the `detections` field is an empty list.

All detections above confidence threshold 0.05 or 0.1 (depending on the version of the API) are recorded in the output file.


##### Classifier outputs

After a classifier is applied, each tuple in a `classifications` list represents `[species, confidence]`. They are listed in order of confidence. The species categories should be interpreted using the `classification_categories` dictionary.  Keys in `classification_categories` will always be nonnegative integers formatted as strings.


## Post-processing tools

The [postprocessing](postprocessing) folder contains tools for working with the output of our detector API.  In particular, [postprocess_batch_results.py](postprocessing/postprocess_batch_results.py) provides visualization and accuracy assessment tools for the output of the batch processing API. A sample output for the Snapshot Serengeti data when using ground-truth annotations can be seen [here](http://dolphinvm.westus2.cloudapp.azure.com/data/snapshot_serengeti/serengeti_val_detections_from_pkl_MDv1_20190528_w_classifications_eval/).


## Integration with other tools

The [integration](integration) folder contains guidelines and postprocessing scripts for using the output of our API in other applications.
