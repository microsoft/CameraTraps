# Overview

[Timelapse](http://saul.cpsc.ucalgary.ca/timelapse/) is an open-source tool for annotating camera trap images.  We have worked with the Timelapse developer to integrate the output of our API into Timelapse, so a user can:

- Select or sort images based on whether they contain people or animals
- View bounding boxes during image annotation (which can speed up review)

This page contains instructions about how to load our API output into Timelapse.  It assumes familiarity with Timelapse, most importantly with the concept of Timlapse templates.


# Download the ML-enabled version of Timelapse

This feature is not in the stable release of Timelapse yet; you can download from (obfuscated URL) or, if you&rsquo;re feeling ambitious, you can build from source on the [machinelearning-experimental](https://github.com/saulgreenberg/Timelapse/tree/machinelearning-experimental) branch of the Timelapse repo.

# Preparing your Timelapse template 

A 