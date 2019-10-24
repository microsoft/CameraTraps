# Overview

We like to think our camera trap detector model is pretty good, but we admit it&rsquo; not perfect: inevitably, we see some &ldquo;false positives&rdquo;, i.e. pesky branches, piles of snow, or roadside litter that our model thinks is an animal.  As with <i>all</i> objection models, you can reduce your false positive rate by raising your detection threshold, but if you raise it too high, you risk missing objects you care about.

One of the things we can take advantage of for camera traps, though, is the fact that cameras typically take thousands of images from the same perspective, and if a detector thinks that branch is an animal in one image, it probably identifies the same branch in <i>hundreds</i> of other images.  If <i>exactly the same bounding box</i> is predicted on many images, we call that a <i>suspicious detection</i>.

Suspicious detections aren&lsquo;t definitely false positives though: sleeping animals can occur in many images without moving an inch, and sometimes cameras on trails frequently have humans entering from the same spot, so we might see thousands of legitiamte detections around that spot, and some of them are bound to be about the same size.

Consequently, we have a set of scripts that:

1. Identifies &ldquo;suspicious detections&rdquo;
2. Makes it easy for a human to very efficiently review just a small fraction of those images to see which ones are really false positives
3. Removes the false positives from a result set

This whole process can eliminate tens of thousands of false detections with just a few minutes of human annotation.

This document shows you how to run these scripts.


# Prerequisites

The suspicious detection elimination process assumes the following:

1. You have checked out this repo and put the repo base on your Python path.  If you are using Windows, for example, you would do this by finding the directory to which you cloned this repo, and adding that directory to your PYTHONPATH environment variable.  Here&rsquo;s a <a href="https://www.computerhope.com/issues/ch000549.htm">good page</a> about editing environment variables in Windows.
2. You have run our our <a href="https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing">batch processing API</a> on our images, and have the .json file it produced for your images.
3. Your images are organized such that the lowest-level folder is a camera.  For example, if you have images in `c:\my_images\2019\B1`, everything in `B1` comes from the same caemra.  This matters because we won&rsquo;t even compare images in this folder to images in `c:\my_images\2019\A1`.  If your images are arranged differently, but there&rsquo;s still some easy way to identify which images are from the same camera, <a href="mailto:cameratraps@microsoft.com">contact us</a>.


# Finding suspicious detections

The first step is to find all the detections that are suspicious, i.e. cases where the same detection is repeated a bunch of times.  For this step, you will use the script:

`(camera trap repo base)/api/batch_processing/postprocessing/find_repeat_detections.py`

This script is going to generate a results file that you probably won&rsquo;t look at (more on this later), and a bunch of images that you will look at, so before running the script, we recommend creating a folder to put all those images.  

So let&rsquo;s assume that:

* Your .json results file is in `c:\my_results.json`
* You want to put the results of this script in a file called `c:\repeat_detection_stuff\results.json`
* You want all the temporary images to end up under `c:\repeat_detection_stuff`
* Your images are in `c:\my_images`

You would run:

`python find_repeat_detections.py "c:\my_results.json" "c:\repeat_detection_stuff\results.json" --imageBase "c:\my_images" --outputBase "c:\repeat_detection_stuff"`

This script can take a while!  Possibly hours if you have millions of images.

There are lots of other options to this script; we&rsquo;ll talk about them later.  They all relate to the things you can do to make the basic process even more efficient by controlling what gets identified as &ldquo;suspicious&rdquo;.


# Cleaning up the suspicious detections that were, in fact, real objects

When the script finishes, you'll have a directory called something like `filtering_2019.10.24.13.40.45` inside the main directory you specified above.  For example, using our running example: 

`c:\repeat_detection_stuff\filtering_2019.10.24.13.40.45`

This directory will have lots of pictures with bounding boxes on them.  Importantly, you are not looking at <i>every</i> detection; each one of these images represents potentially very many nearly-identical detections.  Even though you&rsquo;re doing some manual work here, machine learning is saving you lots of time!

Most of these images indeed correspond to repeated false positives:

<img style="margin-left:50px;" src="images/false_positive.jpg" width="700"><br/>

But some are just animals that aren&rsquo;t moving much:

<img style="margin-left:50px;" src="images/true_positive.jpg" width="700"><br/>

Anything left in this folder will be considered a false positive and removed from your results in subsequent steps<, so the next task is to <i>delete all the images in this folder that have bounding boxes on actual objects of interest</i>.

Note that it&rsquo;s common to have a false positive in an image that also has an animal in it; you can safely delete these, because these scripts operate on individual <i>detections</i>, not <i>images</i>.  So this image is safe to delete:

<img style="margin-left:50px;" src="images/mixed_positive.jpg" width="700"><br/>

Every once in a while you'll see a box that&rsquo;s <i>partially</i> on an animal.  This is <i>probably</i> a false positive that happened to also include an animal, but if it&rsquo;s a close call, the conservative thing to do is always to <i>not</i> delete this image, i.e. leave it in the folder.  Example:

<img style="margin-left:50px;" src="images/possible_positive.jpg" width="700"><br/>

You can do this step (deleting images) using any tool you like, but for the author&rsquo;s two cents, I really like having two windows open:

1. The regular Windows explorer with &ldquo;view&rdquo; set to &ldquo;extra large icons&rdquo;.
2. <a href="https://www.irfanview.com">IrfanView</a>, which is a simple, fast image viewer that makes it very quick to page through lots and lots of images in a row that are all just branches and leaves (by pressing or holding down the &ldquo;right&rdquo; key), and you can just press the &ldquo;delete&rdquo; key in IrfanView to delete an image when you see an animal/person.  This makes things very fast!

Remember that in the next step, we&rsquo;ll be marking any detections left in this folder as false positives, so you probably won&rsquo;t see any of these images again.  <b>So make sure to delete all the images with boxes on stuff you care about!</b>


# Producing the final &ldquo;filtered&rdquo; output file

When that directory contains only false positives, you&rsquo;re ready to remove those - and the many many images of the same detections that you never had to look at - from your results.  To do this, you&rsquo;ll use this script:



# Advanced options
