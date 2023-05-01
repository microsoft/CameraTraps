# Managing camera trap API tasks

## Overview

This document describes the process for running partner data through our <a href="https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/">MegaDetector Batch Processing API</a>.  It assumes that the target data is in a single blob container to which we have a read-write SAS token.

The requirement for write permissions is only used to write intermediate files and API output, so it would be a very small set of code changes to handle the case where we have read-only access to the source container, but for now, we're assuming r/w access.

The major steps covered here are:

* Enumerating the files that need processing and generating API input
* Calling the API, including any necessary resubmissions due to failed shards
* Postprocessing to generate preview files
* Repeat detection elimination

Repeat detection elimination is a manual step that we do in ~30% of cases, and we typically tune the process so this step takes around 20 minutes of hands-on time.  Without this, this whole process should take around five minutes of hands-on time, plus the time required to run the task (which can be anywhere from minutes to days).  I know this looks like a lot of steps, but once you get the hang of it, it's really fast.  If it's your third time doing this and you find that it's taking more than five minutes of human intervention time &ndash; including generating SAS tokens and uploading results for preview &ndash; email cameratraps@lila.science to let us know!

This document is written 98% for internal use, so you will see some instructions that only make sense internally (like "ask Dan to create a password for blah blah").  But if you find it useful externally, let us know!


## Magic strings you need before following this guide

* Storage account and container name for the data container
* API endpoint URL and required "caller" token... for this document, we'll use "blah.endpoint.com" and "caller", respectively.
* Read-only and read-write SAS tokens for the data container... for this document, we'll use "?st=sas_token"
* Credentials for the VM where we host previews and output data... for this document, we'll use "datavm.com".
* A password for the specific folder you will post the results to on that VM
* Possibly a set of specific folders to process as separate tasks within the target container


## Setting up your environment (one time only)

* Unless otherwise stated, you will want to work on a VM in South Central US.  You will not be moving substantial volumes of images, so it's OK to work outside of Azure, but a few steps will be slightly faster with low-latency access.  These instructions will also assume you have a graphical/interactive IDE (Spyder, PyCharm, or VS Code) and that you can run a browser on the same machine where you're running Python.

* Probably install <a href="https://www.postman.com/">Postman</a> for task submission

* If you're working on Windows, probably install <a href="https://www.irfanview.com/">IrfanView</a> for repeat detection elimination (the semi-automated step that will require you to look at lots of images).

* If you're working on Windows, probably install <a href="https://www.bitvise.com/">Bitvise</a> for SCP'ing the results to our Web server VM

* Clone the following repos, and be on master/latest on both:
  * <a href="https://github.com/microsoft/CameraTraps">github.com/microsoft/CameraTraps</a>
  * <a href="https://github.com/microsoft/ai4eutils">github.com/microsoft/ai4eutils</a>

* Put the roots of both of the above repos on your PYTHONPATH; see <a href="https://github.com/microsoft/CameraTraps/#other-notes">instructions on the CameraTraps repo</a> re: setting your PYTHONPATH.

* If you're into using conda environments, cd to the root of the CameraTraps repo and run:

  `conda env create --file environment-api-task-management.yml`


## Stuff you do for each task

### Forking the template script

* Make a copy of <a href="https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/data_preparation/manage_api_submission.py">manage_api_submission.py</a>, <i>outside</i> of the CameraTraps repo.  You may or may not end up with credentials in this file, so your working copy should <i>not be on GitHub</i>.  Name this file as `organization-YYYYMMDD.py`.

* Fill in all the constants in the "constants I set per task" cell.  Specifically:

* storage_account_name
* container_name
* task_set_name, formatted as `organization-YYYYMMDD` (same as the file name)
* base_output_folder_name (this is a local folder... I recommend maintaining a local folder like c:\camera_trap_tasks and putting all task data in subfolders named according to the organization, e.g. c:\camera_trap_tasks\university_of_arendelle, but this isn't critical)
* read_only_sas_token
* read_write_sas_token
* caller
* endpoint_base

If applicable (but usually not applicable):

* container_prefix (restricts image enumeration to specific prefixes in the source container)
* folder_names (splits the overall task up into multiple sub-tasks, typically corresponding to folders that are meaningful to the organization, e.g. "Summer_2018")
* additional_task_args (typically used to specify a model version)


### Preparing the task(s)

I use this file like a notebook, typically running all cells interactively.  The cell notation in this file is friendly to Spyder, VS Code, and PyCharm (professional).  To prepare the task, run all the cells through "generate API calls for each task".

At this point, the json-formatted API string for all tasks (typically just one, unless you used the "folder_names" feature to create multiple tasks), and you're ready to submit.


### Submitting the task(s)

The next cell is called "run the tasks", and though it doesn't actually work, I don't recommend programmatic submission anyway.  You are about to spin up sixteen expensive and power-hungry GPUs, and IMO it's better to do this manually so you can triple-quadruple check that you really want to start a task.  I do this through Postman; see <a href="https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#other-notes-and-example">here</a> for an example.    If you are running multiple tasks, you should run them separately in Postman.

You will get back a task ID for each task, enter these in the "manually define task groups" cell in the format indicated in the template code.  A "task group" is a logical task; the reason we use a <i>list</i> of task IDs for each task group is that (1) we split tasks over 1M images into multiple tasks, and (2) sometimes shards fail and we resubmit some images later as part of the same task, so we will extend those lists as necessary.

I then typically run the "estimate total time" cell.  For very small tasks, this isn't meaningful, since it doesn't include spin-up time.  This tells me when I should check back again.  I then typically run the "status check" cell to confirm the task is in progress.


### Time passes...

Do other work, watch Netflix (Last Kingdom Season 4 just came out!), go to bed, wake up...

When you're back, run the "status check" cell again, and if it doesn't show "completed", wait longer.  If it's been suspiciously long, check in with us.


### Check for failures

Run the "look for failed shards" cell.  Most of the time it will say "no resubmissions necessary".  If it shows some required resubmissions, look carefully at the "missing images" printout.  If it's actually just a small number (but still slightly larger than the `max_tolerable_missing_images` constant, otherwise you wouldn't get this printout), consider just raising the `max_tolerable_missing_images` constant.  This is subjective and project-specific.

If you do have to resubmit tasks, the API calls will be in your console.  Run them, and see the "Resubmit tasks for failed shards" cell, where you need to add the task IDs for the resubmissions to the appropriate task groups.

Theoretically you could have to do all this again if your resubmissions fail, thinking through this is outside the scope of this README.  I've never had this happen.


### Post-processing

Run the next two cells, which should uneventfully pull results and combine results from resubmitted tasks into single .json files.

Now the excitement starts again with the "post-processing" cell: running this will take a minute or two, and browser tabs should open with previews for each task.  I typically decide two things here, both subjective:

1. Do we need to adjust the confidence threshold from the 80% default?

2. Do we need to do the repeat detection elimination step?

The latter isn't just about the results; it's about the priority of the task, the time available, the degree to which the collaborator can do this on their own, etc.  Guidance for these two decisions is beyond the scope of this document.


### Repeat detection elimination (not typically necessary)

Before reading this, I recommend skimming the <a href="https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/postprocessing/repeat_detection_elimination">public documentation on the repeat detection elimination (RDE) process</a>, to get a sense of what it's about.

OK, you're back... I run RDE in the following steps:

1. Run the "repeat detection elimination, phase 1" cell.

2. Before actually starting the manual step, get a rough sense of how many images you have in the target folder.  If it's more than you have time to deal with (I typically aim for no more than ~2k), adjust parameters and re-run this cell.  Also if you see quickly that there are lots of actual true positives with boxes in the output folder (i.e., lots of animals that were just sitting really still), you'll also want Parameter adjustment is also beyond the scope of this document, we'll update in the future with examples for when you might adjust each parameter.

3. Now you're ready to do the manual step, i.e. deleting all the images in the RDE folder with boxes that contain animals.  Reminder: it's fine if the <i>image</i> contains an animal, we're deleting images where you see <i>boxes</i> that contain animals.  <i>Not</i> deleting an image is equivalent to marking it as a false positive in this process, so if you're unsure, it's always safer to delete the image from the RDE folder, which will leave the image in the final output set.  There's rarely harm in deleting a few too many from the RDE folder.

4. For this step, I strongly recommend <a href="https://www.irfanview.com/">IrfanView</a>.  I keep one hand on the "page-down" key and one hand on the "delete" key, and I can blast through several images a second this way.

5. OK, you're back, and you just looked at a lot of images with boxes on trees and other annoying stuff.  Now run the "post-processing (post-RDE)" cell to generate a new HTML preview.  You should see that the number of detections is lower than in the preview you generated earlier, since you just got rid of a bunch of detections.


### Uploading previews to our Web server

For now, ask Dan to create a login and associated folder on our Web server.  If the organization associated with this task is called "university_of_arendelle", Dan will create a folder at `/datadrive/html/data/university_of_arendelle`.  You should copy (with SCP) (I use <a href="https://www.bitvise.com/">Bitvise</a>) the postprocessing folder(s) there, e.g. if your output base was:

`g:\university_of_arendelle`

...and your task set name was:

`university_of_arendelle-20200409`

You will copy a folder that looks like:

`g:\university_of_arendelle\university_of_arendelle-20200409\postprocessing\university_of_arendelle-20200409_0.800`

This will be externally visible (though password-protected) at:

`http://datavm.com/data/university_of_arendelle/university_of_arendelle-20200409_0.800`


### Uploading results to our file share

The .json results files - including the results before and after repeat detection elimination, if applicable - are generally uploaded to our AI for Earth file share when anything somewhat stable is uploaded to the Web server.  This is just a placeholder to add instructions later.  Note to self: we generally zip .json files if they're larger than ~50MB.
