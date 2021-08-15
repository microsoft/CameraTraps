#
# remove_exif.py
#
# Removes all EXIF/IPTC/XMP metadata from a folder of images, without making backup copies, using pyexiv2.
#

#%% Imports and constants

import os
import glob

input_base = r'f:\images'
assert os.path.isdir(input_base)


#%% List files

all_files = [f for f in glob.glob(input_base + "*/**", recursive=True)]
image_files = [s for s in all_files if (s.lower().endswith('.jpg'))]
    

#%% Remove EXIF data (support)

import pyexiv2

# PYEXIV2 IS NOT THREAD SAFE; DO NOT CALL THIS IN PARALLEL FROM A SINGLE PROCESS
def remove_exif(fn):
    
    try:
        img = pyexiv2.Image(fn)
        # data = img.read_exif(); print(data)
        img.clear_exif()
        img.clear_iptc()
        img.clear_xmp()
        img.close()        
    except Exception as e:
        print('EXIF error on {}: {}'.format(fn,str(e)))
    

#%% Debug

if False:    
    #%%
    fn = image_files[-10001]
    os.startfile(fn)    
    #%%
    remove_exif(fn)
    os.startfile(fn)
    
    
#%% Remove EXIF data (execution)

from joblib import Parallel, delayed

n_exif_threads = 50
    
if n_exif_threads == 1:
    
    # fn = image_files[0]
    for fn in image_files:
        remove_exif(fn)
        
else:
    # joblib.Parallel defaults to a process-based backend, but let's be sure
    # results = Parallel(n_jobs=n_exif_threads,verbose=2,prefer='processes')(delayed(remove_exif)(fn) for fn in image_files[0:10])
    results = Parallel(n_jobs=n_exif_threads,verbose=2,prefer='processes')(delayed(remove_exif)(fn) for fn in image_files)

