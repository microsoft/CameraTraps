# What&rsquo;s the deal with AI for Earth camera trap collaborations?

## Table of contents

1. [Overview](#overview)<br/>
2. [Information about our camera trap work](#information-about-our-camera-trap-work)<br/>
3. [Questions about specific camera trap use cases](#questions-about-specific-camera-trap-use-cases)<br/>

## Overview

This page summarizes what we do to help our collaborators, typically ecologists, more specifically ecologists who are overwhelmed by camera trap images.  This page also includes some questions we ask new collaborators, to help assess whether our tools are useful, and &ndash; if so &ndash; what the right set of tools is for a particular project.

Basically this page is the response we give when someone emails us and says &ldquo;I have too many camera trap images!  Can you help me?!?!&rdquo;.  If you&rsquo;re an ecologist reading this page, and that sounds familiar, feel free to answer the questions below in an email to <a href="mailto:cameratraps@microsoft.com">cameratraps@microsoft.com</a>.

## Information about our camera trap work

All of our work in this space is open-sourced here:

&nbsp;&nbsp;&nbsp;&nbsp;<https://github.com/Microsoft/cameratraps>

The core of our work is a model &ndash; we call it the &ldquo;MegaDetector&rdquo; &ndash; that we use to separate images that are empty/animals/people/vehicles:

&nbsp;&nbsp;&nbsp;&nbsp;<https://github.com/microsoft/CameraTraps/blob/master/megadetector.md>

Watch the fun video there and you&rsquo;ll get the idea.  Note that this model does not identify specific species.  We believe there will never be a one-size-fits-all species classifier for the whole world, but at this more generic level, we can help people in lots of ecosystems without training custom models for each project.  

Of course, a machine learning model isn&rsquo;t useful to most ecologists by itself, so we “package” this model in a variety of ways.  For most of our collaborators, they send us images (anywhere from thousands to millions), which we run through this detector in the cloud, then we send back a results file, and we&rsquo;ve integrated with a variety of tools that camera trap researchers already use, to make it relatively painless to use our results in the context of a real workflow.  Our most mature integration is with an open-source tool called <a href="http://saul.cpsc.ucalgary.ca/timelapse/">Timelapse</a>:

&nbsp;&nbsp;&nbsp;&nbsp;<https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/integration/timelapse.md>

In some cases, we also train species classifiers for specific ecosystems, and we have tools for training classifiers (we actually train classifiers on individual animals found by our detector, not on whole images):

&nbsp;&nbsp;&nbsp;&nbsp;<https://github.com/microsoft/CameraTraps/blob/master/classification/TUTORIAL.md>

…but we believe that most people will see much larger efficiency gains at the more generic level, so this is where we have focused so far.

We also package our camera trap model into two different APIs: a batch processing API for processing large volumes of images with some latency, and a real-time API for interactive scenarios (e.g. anti-poaching applications).

A slightly more detailed description of this work is available here, including a list of some of the organizations currently leveraging our tools:

&nbsp;&nbsp;&nbsp;&nbsp;<https://medium.com/microsoftazure/accelerating-biodiversity-surveys-with-azure-machine-learning-9be53f41e674>

Usually the first step with a new collaborator is just running our model on a few thousand images and seeing what happens, so if you&rsquo;re interested in trying this on your images, we can work out a way to transfer a set of example images.

After that, we&rsquo;ll typically send back a page of sample results; depending on whether you already know the &ldquo;right&rdquo; answer for these images, the results will look like one of these:

&nbsp;&nbsp;&nbsp;&nbsp;<http://dolphinvm.westus2.cloudapp.azure.com/data/snapshot-serengeti/s7-eval/postprocessing-no-gt/><br/>
&nbsp;&nbsp;&nbsp;&nbsp;[with no ground truth, i.e. without knowing the right answers]
	
&nbsp;&nbsp;&nbsp;&nbsp;<http://dolphinvm.westus2.cloudapp.azure.com/data/snapshot-serengeti/s7-eval/postprocessing-detection-gt/><br/>
&nbsp;&nbsp;&nbsp;&nbsp;[with ground truth, i.e. where we know the right answers]

In case it&rsquo;s helpful, we also maintain a literature survey on “everything we know about machine learning for camera traps”:

&nbsp;&nbsp;&nbsp;&nbsp;<https://github.com/agentmorris/camera-trap-ml-survey>

…and we also maintain a repository of public, labeled camera trap data to facilitate training new models (the largest such repository that we&rsquo;re aware of):

&nbsp;&nbsp;&nbsp;&nbsp;<http://lila.science/datasets>
	
Our camera trap work is part of our larger efforts in using machine learning for biodiversity monitoring:

&nbsp;&nbsp;&nbsp;&nbsp;<http://aka.ms/biodiversitysurveys>


## Questions about specific camera trap use cases

These questions help us assess how we can best help a new collaborator, and which of our tools will be most applicable to a particular project.

1. Can you provide a short overview of your project?  What ecosystem are you working in, and what are the key species of interest?

2. About how many images do you have that you&rsquo;ve already annotated, from roughly the same environments as the photos you need to process in the future?

3. If you have some images you&rsquo;ve already annotated:

  - Did you keep all the empty images, or only the images with animals?
  - Are they from exactly the same camera locations that you need to process in the future (as in, cameras literally bolted in place), or from similar locations?

4. About how many images do you have waiting for processing right now?

5. About how many images do you expect to process in the next, e.g., 1 year?

6. What tools do you use to process and annotate images?  For example, do you:

  - Move images to folders named by species
  - Keep an Excel spreadsheet open and fill it with filenames and species IDs
  - Use a tool like Timelapse or Reconyx MapView that&rsquo;s specifically for camera traps
  - Use a tool like Adobe Bridge or digiKam that&rsquo;s for general-purpose image management
	
7. About what percentage of your images are empty?

8. About what percentage of your images typically contain vehicles or people, and what do you want to do with those images?  I.e., do you consider those &ldquo;noise&rdquo; (the same as empty images), or do you need those labeled as well?

9. What is your level of fluency in Python?  

10. Do you have a GPU available (or access to cloud-based GPUs)?  &ldquo;I don't know what a GPU is&rdquo; is a perfectly good answer.
