#
# retrieve_sample_image.py
#
# Downloader that retrieves images from Google images, used for verifying taxonomy
# lookups and looking for egregious mismappings (e.g., "snake" being mapped to a fish called
# "snake").
#
# Simple wrapper around simple_image_download, but I've had to swap in and out the underlying
# downloader a few times.
#

#%% Imports and environment

from taxonomy_mapping import simple_image_download
google_image_downloader = simple_image_download.simple_image_download()


#%% Test driver

if False:
    
    paths = google_image_downloader.download(keywords='redunca',output_directory=r'c:\temp\downloads',limit=5)
    print(paths)


#%% Main entry point
    
def download_images(query,output_directory,limit=100,verbose=False):
    paths = google_image_downloader.download(keywords=query,output_directory=output_directory,
                                             limit=limit)
    return paths
