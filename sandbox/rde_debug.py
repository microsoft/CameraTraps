#
# rde_debug.py
#
# Some useful cells for comparing the outputs of the repeat detection 
# elimination process, specifically to make sure that after optimizations,
# results are the same up to ordering.
#

#%% Compare two RDE files

import json
from deepdiff import DeepDiff

f1 = '/home/user/postprocessing/ffi/ffi-2022-02-09/rde_0.60_0.85_10_0.20_task_0/filtering_2022.02.09.14.17.32/detectionIndex.json'
f2 = '/home/user/postprocessing/ffi/ffi-2022-02-09/rde_0.60_0.85_10_0.20_task_0/filtering_2022.02.13.17.53.33/detectionIndex.json'

with open(f1,'r') as f:
    d1 = json.load(f)

with open(f2,'r') as f:
    d2 = json.load(f)

assert len(d1['suspiciousDetections']) == len(d2['suspiciousDetections'])

# i_dir = 0
for i_dir in range(0,len(d1['suspiciousDetections'])):
    
    detections1 = d1['suspiciousDetections'][i_dir]
    detections2 = d2['suspiciousDetections'][i_dir]
    
    if len(detections1) > 0:
        # break
        pass
        
    # Regardless of ordering within a directory, we should have the same
    # number of unique detections
    assert len(detections1) == len(detections2)
    
    # Re-sort
    if len(detections1) > 0:
        detections1 = sorted(detections1, key = lambda i: i['sampleImageRelativeFileName'])
        detections2 = sorted(detections1, key = lambda i: i['sampleImageRelativeFileName'])
        d1['suspiciousDetections'][i_dir] = detections1
        d2['suspiciousDetections'][i_dir] = detections2

    # Make sure that we have the same number of instances for each detection        
    for i_det in range(0,len(detections1)):
        assert len(detections1[i_det]['instances']) == len(detections2[i_det]['instances'])
        
    # Make sure the box values match
    for i_det in range(0,len(detections1)):
        box1 = detections1[i_det]['bbox']
        box2 = detections2[i_det]['bbox']
        assert box1 == box2
        
        
diff = DeepDiff(d1,d2)
print(diff)

