# Uploading images to the cloud for processing with MegaDetector

## Before you read the rest of this page...

Everything in this document assumes we have already talked to you about running MegaDetector on your images.  If you're just getting started and wondering whether MegaDetector is for you, start [here](https://github.com/microsoft/CameraTraps/blob/master/collaborations.md) instead.


## General tips before uploading lots of images

If you are uploading terabytes of images, you want to make sure you're getting the most of your Internet connection <i>before</i> you start uploading.  If it's at all possible to be hard-wired, do that.  If you're on WiFi, get as close as you can to your router before you start uploading.  A great way to estimate your transfer time is to go to [speedtest.net](https://speedtest.net) to estimate your upload bandwidth, then head over to [this transfer time calculator](https://www.expedient.com/knowledgebase/tools-and-calculators/file-transfer-time-calculator/) to punch in your data size and upload bandwidth.  If you're looking at days of upload, that's fine, just plan around it and make sure your computer will stay on and connected.  If you're looking at months of upload, it's probably a good idea to consider whether you really want to upload, vs. transferring via hard drive.


## Uploading images to Azure

In the cases where we run MegaDetector on behalf of a user, we need to get the images from you first.  If you already have them accessible on the cloud, that's great, and we are happy to download from a secure URL on Azure, AWS, GCP, etc.  If you have limited bandwidth, we are also happy to receive a hard drive in the mail.

But most of the time, we create a storage container on Azure (Microsoft's cloud) for our users, and we ask you to upload to the cloud.  Typically, we will send you a storage URL that looks something like this:

`https://cameratrapstuff.blob.core.windows.net/your-organization?sv=2020-10-02&si=your-organization-rw&sr=c&sig=987897asdfadhfa90ds7fasdfkljahdsfsd%3D`

That URL is like your password; it will let you read and write data in your container, and it typically expires after a few weeks.  If you want to learn more about what this funny-looking URL is, it's a [shared access signature (SAS) URL](https://docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview).

There are two ways you can use that URL to upload images to the Microsoft cloud: a command-line tool ([AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)) or a graphical tool ([Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/)).

(No, the URL I used in this example is not real, i.e. I did not just do the storage equivalent of posting my credit card number to Twitter.)


### Uploading images to Azure with AzCopy (command-line)
  
If you are at even a little bit comfortable with a command prompt (Windows, Mac, or Linux), we recommend uploading with [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10), a command-line tool for, among other things, uploading data to the Microsoft cloud.  In fact, if you even have a neighbor who is comfortable with a command prompt, and you can buy her/him a granola bar in exchange for command-line help, we still recommend AzCopy.

First, download AzCopy, and unzip it somewhere.  Let's imagine you unzipped it to `c:\users\dan\azcopy`.

Open a windows command prompt (good instructions [here](https://www.howtogeek.com/235101/10-ways-to-open-the-command-prompt-in-windows-10/) for opening a command prompt), and - in our example - run:

`cd c:\users\dan\azcopy`

Of course, that's not actually the name of the folder where you unzipped AzCopy, so replace the folder name with wherever you unzipped AzCopy.

Now let's imagine you have all of your images in a folder called <span style="background-color:yellow;">c:\my_images</span>, and we gave you the URL above to use for uploading.  You can upload that folder (and all of its subfolders) with:

azcopy cp "<span style="background-color:yellow;">c:\my_images</span>" "<span style="background-color:lightgreen;">https://cameratrapstuff.blob.core.windows.net/your-organization?sv=2020-10-02&si=your-organization-rw&sr=c&sig=987897asdfadhfa90ds7fasdfkljahdsfsd%3D</span>" --recursive

Change the yellow highlighted part to the actual folder where your images are, and the green highlighted part to the URL we gave you for your storage container.  And, voila, your transfer should start.  This command prompt window needs to stay open until the transfer finishes.

AzCopy also includes features to limit the bandwidth it uses and to resume interrupted jobs, <a href="mailto:cameratraps@lila.science">mail us</a> if you need help with those options. 


### Uploading images to Azure with Azure Storage Explorer (GUI)

Our second-favorite choice for helping folks upload images to Azure is [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/), a graphical tool that calls AzCopy in the background.  To upload images with Storage Explorer, first download, install, and run Storage Explorer, then click the little connection icon in the upper-left corner:

<img src="images/ase-connection-icon.png" style="width:100px;margin-left:30px;">

Then click "blob container", and when it asks "how will you connect to the blob container?", click "Shared access signature URL (SAS)"), then click "next":

<img src="images/ase-connection-method.jpg" style="margin-left:30px;">

Under "display name", put anything you want, this is just how the container will appear in your Storage Explorer window.  Let's go with "megadetector-upload" for now.  Paste the URL we gave you in the "blob container SAS URL" box:

<img src="images/ase-sas-url.jpg" style="margin-left:30px;">

Click "next", then click "connect".  A tab will open showing you the container, which is probably empty:

<img src="images/ase-empty-container.jpg" style="margin-left:30px;">

You can drag and drop items here, just realize that when you drop a folder that has 1TB of images in it, you're starting a long operation that may take a few days.  This window needs to stay open until the transfer completes.

When you start Storage Explorer in the future, you can get back to this tab by clicking the the little Morse-code-looking icon in the upper-left, then expanding "local and attached", then "storage accounts", then "blob containers", and you will see your container in the list of blob containers:

<img src="images/ase-container.png" style="width:400px;margin-left:30px;">

Always feel free to <a href="mailto:cameratraps@lila.science">mail us</a> if you get stuck!



