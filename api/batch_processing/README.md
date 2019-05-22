
# Detector batch processing API user guide

We offer a service for processing a large quantity of camera trap images using our [MegaDetector](https://github.com/Microsoft/CameraTraps#megadetector) by calling an API, documented here. You can process a batch of up to 2 million images in one request to the API. If in addition you have some images that are labeled, we can evaluate the performance of the MegaDetector on your labeled images (not documented here).

All references to "container" in this document refer to [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/) containers. 

## 1. Processing time

It takes about 0.8 seconds per image per machine, and we have at most 16 machines that can process them in parallel. So if no one else is using the service and you'd like to process 1 million images, it will take 1,000,000 * 0.8 / (16 * 60 * 60) = 14 hours. 


## 2. API

### 2.1 Endpoints

The endpoints of this API are available at

```
http://URL/v1/camera-trap/detection-batch
```

#### `/request_detections`
To submit a request for batch processing, make a POST call to

```http://URL/v1/camera-trap/detection-batch/request_detections```.

with a json body containing input fields defined below. The API will return with a json response very quickly to give you a RequestID representing the request you have submitted (or an error message, if your inputs are not acceptable), for example:
```json
{
  "request_id": "13H8A43FDE"
}
```

#### `/task`
Check the status of your request by calling the `/task` endpoint via a GET call, passing in your RequestID:

```http://URL/v1/camera-trap/detection-batch/task/RequestID```

This returns a json with the fields `status`, `uuid`, and a few others. The `status` field contains a stringfield json object with the following fields: 

- `request_status`: one of `running`, `failed`, `problem`, and `completed`. The status `failed` indicates that the images have not been submitted to the cluster for processing, and so you can go ahead and call the endpoint again, correcting your inputs according to the message shown. The status `problem` indicates that the images have already been submitted for processing but the API encountered an error while monitoring progress; in this case, *please do not retry*, contact us to retrieve your results so that abandoned jobs are not using up resources. 

- `message`: a longer string describing the request_status and any errors; when the request is completed, the URLs to the output files will also be here (see Outputs section below).

- `time`: timestamp in UTC time.


#### `/supported_model_versions`
Check which versions of the MegaDetector are supported by this API by making a GET call to 

```http://URL/v1/camera-trap/detection-batch/supported_model_versions```


#### `/default_model_version`
Check which versions of the MegaDetector is used by default by making a GET call to

```http://URL/v1/camera-trap/detection-batch/default_model_version```


#### Canceling a request
Not yet supported. 

Meanwhile, once the shards of images are submitted for processing, please do not retry if a subsequent call to the `/task` endpoint indicates that there has been a problem. Instead, contact us to retrieve any results. In this case, the `/task` endpoint will return a message object where the `message` field mentions "please contact us", and the `request_status` field is "problem").


### 2.2 Inputs

| Parameter                | Is required | Type | Explanation                 |
|--------------------------|-------------|-------|----------------------------|
| input_container_sas      | Yes         | string | SAS URL with list and read permissions to the Blob Storage container where the images are stored. |
| images_requested_json_sas | No          | string | SAS URL with list and read permissions to a json file in Blob Storage. The json contains a list, where each item (a string) in the list is the full path to an image from the root of the container. An example of the content of this file: `["Season1/Location1/Camera1/image1.jpg", "Season1/Location1/Camera1/image2.jpg"]`.  Only images whose paths are listed here will be processed. |
| image_path_prefix        | No          | string | Only process images whose full path starts with `image_path_prefix` (case-_sensitive_). Note that any image paths specified in `images_requested_json_sas` will need to be the full path from the root of the container, regardless whether `image_path_prefix` is provided. |
| first_n                  | No          | int | Only process the first `first_n` images. Order of images is not guaranteed, but is likely to be alphabetical. Set this to a small number to avoid taking time to fully list all images in the blob (about 15 minutes for 1 million images) if you just want to try this API. |
| sample_n                | No          |int | Randomly select `sample_n` images to process. |
| model_version           | No          |string | Version of the MegaDetector model to use. Default is the most updated stable version (check using the `/default_model_version` endpoint). Supported versions can be listed by calling the `/supported_model_versions` endpoint.|
| request_name            | No          |string | A string (letters, digits, `_`, `-` allowed, max length 32 characters) that will be appended to the output file names to help you identify the resulting files. A timestamp in UTC ("%Y%m%d%H%M%S") of time of submission will be appended to the resulting files automatically. |

- We assume that all images you would like to process in this batch are uploaded to a container in Azure Blob Storage. 
- Only images with file name ending in '.jpg' or '.jpeg' (case insensitive) will be processed, so please make sure the file names are compliant before you upload them to the container (you cannot rename a blob without copying it entirely once it is in Blob Storage). 
- The path to the images in blob storage cannot contain commas (this would confuse the output CSV).

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
  "image_path_prefix": "Alberta_location1_2019",
  "first_n": 100000
}
```

You can manually call the API using applications such as Postman:

![Screenshot of Azure Storage Explorer used for generating SAS tokens with read and list permissions](./documentation/Postman_screenshot.png)


#### How to obtain a SAS token

You can easily generate a [SAS token](https://docs.microsoft.com/en-us/azure/storage/common/storage-dotnet-shared-access-signature-part-1) to a container or a particular blob (a file in blob storage) using the desktop app [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/) (available on Windows, macOS and Linux). You can also issue SAS tokens programmatically by using the [Azure Storage SDK for Python](https://azure-storage.readthedocs.io/ref/azure.storage.blob.baseblobservice.html#azure.storage.blob.baseblobservice.BaseBlobService.generate_blob_shared_access_signature).


Using Storage Explorer, right click on the container or blob you'd like to grant access for, and choose "Get Shared Access Signature...". On the dialogue window that appears, 
- cross out the "Start time" field if you will be using the SAS token right away
- set the "Expiry time" to a date in the future, about a month ahead is reasonable. The SAS token needs to be valid for the duration of this the batch processing request. 
- make sure "Read" and "List" are checked under "Permissions" (see screenshot) 

Click "Create", and the "URL" field on the next screen is the value required for `input_container_sas` or `images_requested_json_sas`. 

![Screenshot of Azure Storage Explorer used for generating SAS tokens with read and list permissions](./documentation/SAS_screenshot.png)


### 2.3 Outputs

Once your request is submitted and parameters validated, the API divides all images into shards of about 2000 images each, and send them to an Azure Machine Learning compute cluster for processing. Another process will monitor how many shards have been evaluated, checking every 30 minutes, and update the status of the request, which you can check via the `/task` endpoint. 

When all shards have finished processing, the `status` returned by the `/task` endpoint will have a `message` field containing a string that can be loaded as a json, with 3 fields each containing an URL to a downloadable file. The `message` field looks like

```json
{
    "uuid": 3821,
    "status": {
        "request_status": "completed",
        "time": "2019-05-22 00:31:51",
        "message": "Completed at 2019-05-22 00:31:51. Number of failed shards: 0. URLs to output files: {'detections': 'https://cameratrap.blob.core.windows.net/async-api-v3-2/3821/3821_detections__20190522002119.json?se=2019-06-05T00%3A31%3A51Z&sp=r&sv=2018-03-28&sr=b&sig=hYWcHrnMbZ8EjQ1t4Rmtx0Ay/DZDa%2BsQehBP4/nySko%3D', 'failed_images': 'https://cameratrap.blob.core.windows.net/async-api-v3-2/3821/3821_failed_images__20190522002119.json?se=2019-06-05T00%3A31%3A51Z&sp=r&sv=2018-03-28&sr=b&sig=xwoi9tFD9pKhAKdoEwx%2BsnS5gRpEE5x3hR1IY4Jll2Y%3D', 'images': 'https://cameratrap.blob.core.windows.net/async-api-v3-2/3821/3821_images.json?se=2019-06-05T00%3A31%3A51Z&sp=r&sv=2018-03-28&sr=b&sig=llDBCWK%2B%2BQHae5rK7U8RchjPN/DZYb96XHB0r/yX8LU%3D'}"
    },
    "timestamp": "2019-05-22 00:21:19",
    "endpoint": "uri"
}
```
 which you can parse to obtain the URLs:
```python
import json

task_status = body['status']
assert task_status['request_status'] == 'completed'
output_files_str = task_status['message'].split('URLs to output files: ')[1]
output_files = json.loads(output_files_str)
url_to_result = output_files['detections']
url_to_failed_images = output_files['failed_images']
url_to_all_images_processed = output_files['images']

```

These URLs are valid for 14 days from the time the request has finished. If you neglected to retrieve them before the links expired, contact us with the RequestID and we can send the results to you. Here are the 3 files to expect:

| File name                | Description | 
|--------------------------|-------------|
| RequestID_detections.csv | Contains the result produced by the detector. It is a table with 3 columns (see below for explanation).   |
| RequestID_failed_images.csv | Contains full paths to images in the blob that the API failed to open, possibly because they are corrupted, or failed to apply the detector model to. |
| RequestID_images.json | Contains a list of the full paths to all images that the API was supposed to process, based on the content of the container at the time the API was called and the filtering parameters provided. |

#### How to interpret the results

The output of the detector is saved in `RequestID_detections.csv`. It looks like

| image_path | max_confidence | detections | 
|------------|----------------|------------|
| folder/subfolders/image1.JPG | 0.9960 | "[[0.5252, 0.1727, 0.9546, 0.43948, 0.9960], [0.8804, 0.4575, 0.94537, 0.5313, 0.1468]]" |
| folder/subfolders/image2.JPG | 0.0 | [] |
| folder/subfolders/image3.jpg | 0.0 | [] |
| folder/subfolders/image4.jpg | 0.4091 | "[[0.2823, 0.1759, 0.3608, 0.2458, 0.4091]]" |

The first column contains the full path to the image in the blob container. 

The second column is the confidence value of the most confident detection on the image (all detections above confidence 0.05 are included so you can select a confidence threshold for determining empty from non-empty).

The third column contains details of the detections so you can visualize them. It is a stringfied json of a list of lists, representing the detections made on that image. Each detection list has the coordinates of the bounding box surrounding the detection, followed by its confidence:

```
[ymin, xmin, ymax, xmax, confidence, (class)]
```

where `(xmin, ymin)` is the upper-left corner of the detection bounding box. The coordinates are relative to the height and width of the image. 

An integer `class` comes after `confidence` in versions of the API that uses MegaDetector version 3 or later. The `class` label corresponds to the following:

```
1: animal
2: person
4: vehicle
```

Note that the `vehicle` class (available in Mega Detector version 4 or later) is number 4. Class number 3 (group) is not included in training and should be ignored (and so should any other class labels not listed here) if it shows up in the result.

When the detector model detects no animal (or person or vehicle), the confidence is shown as 0.0 (not confident that there is an object of interest) and the detection column is an empty list.


All detections above confidence threshold 0.05 are recorded in the output file.


## 3. Post-processing tools

[postprocess_batch_results.py](postprocess_batch_results.py) provides visualization and accuracy assessment tools for the output of the batch processing API.

