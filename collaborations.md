# Getting started with MegaDetector

## Table of contents

1. [Overview](#overview)<br/>
2. [How people run MegaDetector](#how-people-run-megadetector)<br/>
3. [How people use MegaDetector results](#how-people-use-megadetector-results)<br/>
4. [Questions about specific camera trap use cases](#questions-about-specific-camera-trap-use-cases)<br/>
5. [Learn more](#learn-more)<br/>

## Overview

Conservation biologists invest a huge amount of time reviewing camera trap images, and &ndash; even worse &ndash; a huge fraction of that time is spent reviewing images they aren't interested in.  This primarily includes empty images, but for many projects, images of people and vehicles are also "noise", or at least need to be handled separately from animals.

*Machine learning can accelerate this process, letting biologists spend their time on the images that matter.*

To this end, we've trained an AI model &ndash; called "MegaDetector" &ndash; to detect animals, people, and vehicles in camera trap images.  It does not identify animals, it just finds them.  We've also done a little bit of work on training species classifiers, but 99% of what we do is related to MegaDetector.

This page summarizes what we do with that model to help our collaborators, typically ecologists, more specifically ecologists who are overwhelmed by camera trap images.  This page also includes some questions we ask new collaborators, to help assess whether our tools are useful, and &ndash; if so &ndash; what the right set of tools is for a particular project.

Basically this page is the response we give when someone emails us and says "I have too many camera trap images!  Can you help me?!?!".  If you're an ecologist reading this page, and that sounds familiar, feel free to answer the questions below in an email to <a href="mailto:cameratraps@lila.science">cameratraps@lila.science</a>.

You can see a list of some of the organizations who have used our tools [here](https://github.com/microsoft/CameraTraps/#who-is-using-megadetector).

If you are looking for a more technical description of our MegaDetector model, see [this page](megadetector.md).

## How people run MegaDetector

MegaDetector is a publicly-available model, and there are instructions [here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md#using-the-model) for running it using our Python scripts.  Many of our users run MegaDetector on their own, either on the cloud or on their local computers.

That said, we know that Python can be a bit daunting, and that it requires significant processing power to run MegaDetector on millions of images.  So many of our users - particularly high-volume users - send us images (anywhere from tens of thousands to millions), which we run through MegaDetector, then we send back a results file.

Whether you're going to run MegaDetector on your own or work with us, usually the first step with a new user is just running our model on a few thousand images and seeing what happens, so if you're interested in trying this on your images, we can work out a way to transfer a set of example images, just email us at <a href="mailto:cameratraps@lila.science">cameratraps@lila.science</a>.After that, we'll typically send back a page of sample results, like this one:

<https://lila.science/private/snapshot-safari-public/snapshot-safari-2022-02-07_rde_0.60_0.85_10_0.20_0.800>

That page isn't <i>quite</i> what a real results page would look like: rather than just "detections" and "non-detections", a real results page would have images broken out into separate links for empty/people/vehicle/animal.  But we can't use a data set with people in it for a public demo, so the samples above are simplified to just include images that have animals or are empty (that page is based on public data from the <a href="https://www.zooniverse.org/organizations/meredithspalmer/snapshot-safari">Snapshot Safari</a> project.


## How people use MegaDetector results

Of course, running MegaDetector doesn't do anything useful by itself: it just produces a file that tells you which images MegaDetector thinks have animals/people/vehicles in them.  You still need a way to use that file in a real image processing workflow.  We've integrated with a variety of tools that camera trap researchers already use, to make it relatively painless to use our results in the context of a real workflow.  Our most mature integration is with <a href="http://saul.cpsc.ucalgary.ca/timelapse/">Timelapse</a>, a fantastic open-source tool for reviewing camera trap images (very efficient even if you're not using AI!).  Read more about how to use MegaDetector results with Timelapse [here](https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/integration/timelapse.md).

We have somewhat-less-complete integrations with the [eMammal desktop application](https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/integration/eMammal) and with [digiKam](https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/integration/digiKam).

We also have Python tools that use MegaDetector results to just separate a folder of images into folders containing images that are probably-empty, probably-animal, etc., preserving the original folder structure within these folders.  Users often use this approach to just get rid of the images that MegaDetector is really sure are empty, then you can go about your workflow exactly as you did before, just with fewer empty images.


## Questions about specific camera trap use cases

These questions help us assess how we can best help a new collaborator, and which of our tools will be most applicable to a particular project.

1. Can you provide a short overview of your project?  What ecosystem are you working in, and what are the key species of interest?

2. About how many images do you have that you've already annotated, from roughly the same environments as the photos you need to process in the future?

3. If you have some images you've already annotated:

  - Did you keep all the empty images, or only the images with animals?
  - Are they from exactly the same camera locations that you need to process in the future (as in, cameras literally bolted in place), or from similar locations?

4. About how many images do you have waiting for processing right now?

5. About how many images do you expect to process in the next, e.g., 1 year?

6. What tools do you use to process and annotate images?  For example, do you:

  - Move images to folders named by species
  - Keep an Excel spreadsheet open and fill it with filenames and species IDs
  - Use a tool like Timelapse or Reconyx MapView that's specifically for camera traps
  - Use a tool like Adobe Bridge or digiKam that's for general-purpose image management
	
7. About what percentage of your images are empty?

8. About what percentage of your images typically contain vehicles or people, and what do you want to do with those images?  I.e., do you consider those "noise" (the same as empty images), or do you need those labeled as well?

9. What is your level of fluency in Python?  

10. Do you have a GPU available (or access to cloud-based GPUs)?  "I don't know what a GPU is" is a perfectly good answer.

11. How did you hear about MegaDetector?

## Learn more

* We maintain a literature survey on "[everything we know about machine learning for camera traps](https://github.com/agentmorris/camera-trap-ml-survey)".
* We maintain a repository of public, labeled camera trap data to facilitate training new models (the largest such repository that we're aware of) at [lila.science](http://lila.science/datasets).
* Our camera trap work is part of our larger efforts in [using machine learning for biodiversity monitoring](http://aka.ms/biodiversitysurveys).
