# Overview

[Timelapse](http://saul.cpsc.ucalgary.ca/timelapse/) is an open-source tool for annotating camera trap images.  We have worked with the Timelapse developer to integrate the output of our API into Timelapse, so a user can:

- Select or sort images based on whether they contain people or animals
- View bounding boxes during image annotation (which can speed up review... but mostly just looks pretty, the important part is selection)


# Setting Timelapse up to work with our API output

This page used to host long and complicated instructions about loading the output of our Camera Trap API into a test version of Timelapse, but now it's all nicely integrated into Timelapse, so instead of listing lots of stuff here, I&rdquo;ll just tell you to:

- Download Timelapse from [here](http://saul.cpsc.ucalgary.ca/timelapse/pmwiki.php?n=Main.Download2)
- Download the Timelapse User Guide [here](http://saul.cpsc.ucalgary.ca/timelapse/pmwiki.php?n=Main.UserGuide), and check out the section called &ldquo;Automatic Image Recognition&rdquo;


# Do useful stuff with your ML results!

Now that you&rsquo;ve loaded ML results, there are two major differences in your Timelapse workflow... first, and most obvious, there are bounding boxes around animals:

<img src="images/tl_boxes.jpg">

<br/>This is fun; we love both animals and bounding boxes.  But far more important is the fact that you can select images based on whether they contain animals.  We recommend something like the following workflow:

## Confidence level selection

Find the confidence threshold that you&rsquo;re comfortable using to discard images, by choosing select &rarr; custom selection &rarr; confidence < [some number].  0.6 is a decent starting point.  Note that you need to type 0.6, rather than .6, i.e. <i>numbers other than 1.0 need to include a leading zero</i>.

<img src="images/tl_confidence.jpg">

<br/>Now you should only be seeing images with no animals... if you see animals, something is amiss.  You can use the &ldquo;play forward quickly&rdquo; button to very rapidly assess whether there are animals hiding here.  If you&rsquo;re feeling comfortable...

## Labeling

Change the selection to confidence >= [your threshold].  Now you should be seeing mostly images with animals, though you probably set that threshold low enough that you&rsquo;re still seeing <i>some</i> empty images.  At this point, go about your normal Timelapse business, without wasting all that time on empty images!


## Handling images containing people

Many workflows also benefit from quickly identifying images of people, either because they're irrelevant to the survey project or because they need to be removed for compliance reasons.  Because our detector has classes for both people and animals,  separating out all the people &ndash; with the ability to quickly review images near the confidence boundary &ndash; is efficient.  See the Timelapse user's guide for suggested workflows.





