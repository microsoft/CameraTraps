# Output Manager App

The Output Manager is a Windows desktop application for making the batch processing API's output JSON file more manageable in downstream processing. It does either or both of the following:

- Retrieves all result entries (each result entry corresponds to one image, containing all detections on that image) where the image file path matches a specified query string. It optionally replaces that query string with a replacement token. If the query string is blank, it can be used to prepend a token to all image file paths. 

- Splits the API's output JSON file into smaller files each containing only results corresponding to a subfolder of images. 
    - This could be useful for distributing the subsequent manual labeling and verification effort, or loading only the relevant results for a [Timelapse](../integration/timelapse.md) project.
    - All images in the subfolder `blah\foo\bar` will end up in a JSON file called `blah_foo_bar.json`.

The app is functionally the same as [subset_json_detector_output.py](./subset_json_detector_output.py).


## Download

Download the application <a href="https://lilablobssc.blob.core.windows.net/models/apps/CameraTrapApiOutputManager.1.1.zip">here</a>.


## User guide

The app has one interface:

<br/>
<img src="images/CameraTrapJsonManagerApp.jpg" width=600 alt="Screenshot of the Output Manager app, which only has one interface">
<br/>


### Options explained:

| Option                   | Explanation                |
|--------------------------|----------------------------|
| Input file               | Path to the batch processing API's output file. It should end in `.json`. |
| Output file / folder     | Specify a file name (ends with `.json`) if you're replacing parts of image file paths in the output file with another token; specify a directory if creating smaller JSON files each with results for a subfolder of images.
| Query                    | Retrieve result entries with image file path containing this query string/token. Leave blank to retrieve all entries. <br/> <br/> Example: <br/>Specify `Unprocessed Images/Camera 3/` to restrict to images from this folder. <br/>Specify `Location 1` to retrieve all image files that have `Location 1` in its path. <br/>Remember to use forward slash `/` here since it is used in the API's output file. <br/>Regular expression is not supported.
| Replacement              | A string/token to replace the Query string in the image file paths. If Query is left blank, the Replacement string will be prepended to all image file paths.
| Confidence threshold     | Only detections with confidence above this threshold will be copied over.
| Split folders            | Check to split the API's output file.
| Split folder mode        | Choose from “Top”, “Bottom”, "NFromTop" and “NFromBottom”, explained below. |
| Split parameter          | Used to specify `N` if "Split folder mode" is "NFromTop" or "NFromBottom". |
| Copy jsons to folders    | If "Split folders" and "Make folder-relative" are checked, copy each resulting small JSON file to the subfolder containing the corresponding images. |
| Create folders           | If "Copy jsons to folders" is checked, checking this option will create the subfolder in the image directory if it does not yet exist. | 
| Overwrite json files     | Overwrite output files if they already exist. |
| Make folder-relative     | Make the image file paths in the resulting JSON(s) relative to their containing folder. Only meaningful if "Split folders" is checked. |


### "Split folder mode" explained:

Let's say the image file paths in your API output look like this:

```
Unprocessed Images/NE4311/Camera_4/C8_11.11.19/DCIM/103RECNX/RCNX1111.JPG
Unprocessed Images/NE2300/Camera_220/C126_12.12.19/DCIM/101RECNX/RCNX0678.JPG
```

#### Top

If you set “Split folder mode” to “Top”, that means “split image entries into JSON files based on the top folder”, which in this case is just `Unprocessed Images` in our example. This makes more sense if you have top-level folders like, `Summer 2018` and `Winter 2018` and want to divide up the work that way.


#### Bottom

If you set “Split folder mode” to “Bottom”, the app would create JSON files for the bottom-level folders, for example these two folders:
 
```
Unprocessed Images/NE4311/Camera_4/C8_11.11.19/DCIM/103RECNX
Unprocessed Images/NE2300/Camera_220/C126_12.12.19/DCIM/101RECNX
```

This is useful if you have one camera’s data in each bottom-most folder.


#### NFromTop

What if you want each JSON file to represent a subfolder that's not a top-most nor a bottom-most folder?

If all of them are a certain number of levels (`N`) below the top-most folder, you can choose the "NFromTop" option and specify `N` in the "Split parameter" text box. `N` is "how many folders _down from the top_". 

`N` of "0" would be the same as "Top" (top-most folders), "1" would be one more level down, and "2" would result in one JSON for each of
```
Unprocessed Images/NE4311/Camera_4
Unprocessed Images/NE2300/Camera_220
```

#### NFromBottom

Same idea for "NFromBottom", where `N` is "how many folders _up from the bottom_". 

Here `N` of "0" would be the same as "Bottom" (bottom-most folders), "1" would be one level up, and "2" would result in one JSON for each of 
```
Unprocessed Images/NE4311/Camera_4/C8_11.11.19
Unprocessed Images/NE2300/Camera_220/C126_12.12.19
```


### Once you filled in the options


Hit `Process` to proceed. The output window will show a progress bar and the paths to the resulting, smaller JSON files. 

The `Help` button will bring you to this page.



## Other notes

You should not need to worry about the fact that the output file uses forward slashes `/`, except in specifying the Query and Replacement string/token. Both the Output Manager and Timelapse will handle the path separator correctly for the platform you're running on.



## Help

If you run into any issues whatsoever, email us at cameratraps@microsoft.com for help!
