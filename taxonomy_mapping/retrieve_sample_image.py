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

method = 'simple_image_download' # 'google_images_download'

if method == 'simple_image_download':
    
    from taxonomy_mapping import simple_image_download
    google_image_downloader = simple_image_download.Downloader()
    google_image_downloader.directory = r'g:\temp\downloads'
 
elif method == 'google_images_download':
    
    from google_images_download import google_images_download

else:
    
    raise ValueError('Unrecognized method {}'.format(method))

#%%


#%% Main entry point

def download_images(query,output_directory,limit=100,verbose=False):

    query = query.replace(' ','+')        
    
    if method == 'simple_image_download':
        
        google_image_downloader.directory = output_directory
        paths = google_image_downloader.download(query, limit=limit,
          verbose=verbose, cache=False, download_cache=False)
        return paths
        
    elif method == 'google_images_download':
        
        response = google_images_download.googleimagesdownload()    
        arguments = {'keywords':query,'limit':limit,'print_urls':verbose,
                     'image-directory':output_directory}
        response.download(arguments)
        return None

    else:
        
        raise ValueError('Unrecognized method {}'.format(method))
        

#%% Test driver

if False:
    
    #%%
    
    paths = download_images(query='redunca',output_directory=r'g:\temp\download-test',
                    limit=20,verbose=True) 
