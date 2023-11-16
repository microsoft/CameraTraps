#
# idaho-camera-traps.py
#
# Prepare the Idaho Camera Traps dataset for release on LILA.
#

#%% Imports and constants

import json
import os
import numpy as np
import dateutil 
import pandas as pd
import datetime
import shutil

from tqdm import tqdm
from bson import json_util

from collections import defaultdict

# Multi-threading for .csv file comparison and image existence validation
from multiprocessing.pool import Pool as Pool
from multiprocessing.pool import ThreadPool as ThreadPool
n_threads = 14
n_threads_file_copy = 20

input_base = r'i:\idfg-images'
output_base = r'h:\idaho-camera-traps'
output_image_base = r'j:\idaho-camera-traps-output'
assert os.path.isdir(input_base)
assert os.path.isdir(output_base)
assert os.path.isdir(output_image_base)

output_image_base_public = os.path.join(output_image_base,'public')
output_image_base_private = os.path.join(output_image_base,'private')

# We are going to map the original filenames/locations to obfuscated strings, but once
# we've done that, we will re-use the mappings every time we run this script. 
force_generate_mappings = False

# This is the file to which mappings get saved
id_mapping_file = os.path.join(output_base,'id_mapping.json')

# The maximum time (in seconds) between images within which two images are considered the 
# same sequence.
max_gap_within_sequence = 30

# This is a two-column file, where each line is [string in the original metadata],[category name we want to map it to]
category_mapping_file = os.path.join(output_base,'category_mapping.csv')

# The output file, using the original strings
output_json_original_strings = os.path.join(output_base,'idaho-camera-traps-original-strings.json')

# The output file, using obfuscated strings for everything but filenamed
output_json_remapped_ids = os.path.join(output_base,'idaho-camera-traps-remapped-ids.json')

# The output file, using obfuscated strings and obfuscated filenames
output_json = os.path.join(output_base,'idaho-camera-traps.json')

# One time only, I ran MegaDetector on the whole dataset...
megadetector_results_file = r'H:\idaho-camera-traps\idfg-2021-07-26idaho-camera-traps_detections.json'

# ...then set aside any images that *may* have contained humans that had not already been
# annotated as such.  Those went in this folder...
human_review_folder = os.path.join(output_base,'human_review')

# ...and the ones that *actually* had humans (identified via manual review) got
# copied to this folder...
human_review_selection_folder = os.path.join(output_base,'human_review_selections')

# ...which was enumerated to this text file, which is a manually-curated list of
# images that were flagged as human.
human_review_list = os.path.join(output_base,'human_flagged_images.txt')

# Unopinionated .json conversion of the .csv metadata
sequence_info_cache = os.path.join(output_base,'sequence_info.json')

valid_opstates = ['normal','maintenance','snow on lens','foggy lens','foggy weather',
                  'malfunction','misdirected','snow on lense','poop/slobber','sun','tilted','vegetation obstruction']
opstate_mappings = {'snow on lense':'snow on lens','poop/slobber':'lens obscured','maintenance':'human'}

survey_species_presence_columns = ['elkpresent','deerpresent','prongpresent']

presence_to_count_columns = {
    'otherpresent':['MooseAntlerless','MooseCalf','MooseOther','MooseBull','MooseUnkn',
                    'BlackBearAdult','BlackBearCub','LionAdult','LionKitten','WolfAdult',
                    'WolfPup','CattleCow','CattleCalf','other'],
    'elkpresent':['ElkSpike','ElkAntlerless','ElkCalf','ElkRaghorn','ElkMatBull','ElkUnkn','ElkPedNub'],
    'deerpresent':['MDbuck','MDantlerless','MDfawn','WTDbuck','WTDantlerless','WTDfawn','WTDunkn','MDunkn'],
    'prongpresent':['PronghornBuck','PronghornFawn','PHunkn']
    }

required_columns = ['File','Folder','Date','Time','otherpresent','other','otherwhat','opstate']
expected_presence_columns = ['elkpresent','deerpresent','prongpresent','humanpresent','otherpresent']
    
expected_count_columns = set()
for presence_column in presence_to_count_columns.keys():
    count_columns = presence_to_count_columns[presence_column]
    for count_column in count_columns:
        expected_count_columns.add(count_column)
        
def list_is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))


#%% List files (images + .csv)

def get_files():
    
    all_files_list = os.path.join(output_base,'all_files.json')
    force_file_enumeration = False
    
    if (os.path.isfile(all_files_list) and (not force_file_enumeration)):
        
        print('File list exists, bypassing enumeration')
        with open(all_files_list,'r') as f:
            all_files = json.load(f)
            
    else:
        
        from pathlib import Path
        all_files = []
        for path in Path(input_base).rglob('*.*'):
            path = str(path)
            path = os.path.relpath(path,input_base)
            all_files.append(path)
        with open(all_files_list,'w') as f:
            json.dump(all_files,f,indent=1)
    
    print('Enumerated {} files'.format(len(all_files)))
    
    image_files = [s for s in all_files if (s.lower().endswith('.jpg') or s.lower().endswith('.jpeg'))]
    csv_files = [s for s in all_files if (\
                                          (s.lower().endswith('.csv')) and \
                                          ('Backups' not in s) and \
                                          ('Metadata.csv' not in s) and \
                                          ('ExportedDataFiles' not in s) and \
                                          ('CSV Files' not in s)
                                              )]
    
    print('{} image files, {} .csv files'.format(len(image_files),len(csv_files)))
    
    # Ignore .csv files in folders with multiple .csv files
    
    # ...which would require some extra work to decipher.
    
    csv_files_to_ignore = []
    
    folder_to_csv_files = defaultdict(list)
    
    # fn = csv_files[0]
    for fn in csv_files:
        folder_name = os.path.dirname(fn)
        folder_to_csv_files[folder_name].append(fn)
    
    for folder_name in folder_to_csv_files.keys():
        if len(folder_to_csv_files[folder_name]) > 1:
            print('Multiple .csv files for {}:'.format(folder_name))
            for csv_file in folder_to_csv_files[folder_name]:
                print(csv_file)
                csv_files_to_ignore.append(csv_file)
            print('')
            
    n_csv_original = len(csv_files)
    csv_files = [s for s in csv_files if s not in csv_files_to_ignore]
    
    print('Processing {} of {} csv files'.format(len(csv_files),n_csv_original))

    return image_files,csv_files


#%% Parse each .csv file into sequences (function)

# csv_file = csv_files[-1]
def csv_to_sequences(csv_file):
    
    print('Processing {}'.format(csv_file))
    
    csv_file_absolute = os.path.join(input_base,csv_file)
    # os.startfile(csv_file_absolute)
    
    sequences = []
    # survey = csv_file.split('\\')[0]

    # Sample paths from which we need to derive locations:
    #
    # St.Joe_elk\AM99\Trip 1\100RECNX\TimelapseData.csv
    # Beaverhead_elk\AM34\Trip 1\100RECNX\TimelapseData.csv
    #
    # ClearCreek_mustelids\Winter2015-16\FS-001-P\FS-001-P.csv
    # ClearCreek_mustelids\Summer2015\FS-001\FS-001.csv
    # ClearCreek_mustelids\Summer2016\IDFG-016\IDFG-016.csv
    #
    # I:\idfg-images\ClearCreek_mustelids\Summer2016\IDFG-017b
    # I:\idfg-images\ClearCreek_mustelids\Summer2016\IDFG-017a
    if 'St.Joe_elk' in csv_file or 'Beaverhead_elk' in csv_file:
        location_name = '_'.join(csv_file.split('\\')[0:2]).replace(' ','')
    else:
        assert 'ClearCreek_mustelids' in csv_file
        tokens = csv_file.split('\\') 
        assert 'FS-' in tokens[2] or 'IDFG-' in tokens[2]
        location_name = '_'.join([tokens[0],tokens[2]]).replace('-P','')
        if location_name.endswith('017a') or location_name.endswith('017b'):
            location_name = location_name[:-1]
    
    # Load .csv file
    df = pd.read_csv(csv_file_absolute)
    df['datetime'] = None
    df['seq_id'] = None
    df['synthetic_frame_number'] = None
    
    # Validate the opstate column
    opstates = set(df['opstate'])
    for s in opstates:
        if isinstance(s,str):
            s = s.strip()
            if len(s) > 0:
                assert s in valid_opstates,'Invalid opstate: {}'.format(s)
    
    column_names = list(df.columns)
    
    for s in required_columns:
        assert s in column_names
    
    count_columns = [s for s in column_names if s in expected_count_columns]
    
    presence_columns = [s for s in column_names if s.endswith('present')]
    
    for s in presence_columns:
        if s not in expected_presence_columns:
            assert 'Unexpected presence column {} in {}'.format(s,csv_file)
    for s in expected_presence_columns:
        if s not in presence_columns:
            assert 'Missing presence column {} in {}'.format(s,csv_file)
    
    if False:
        for s in expected_count_columns:
            if s not in count_columns:
                print('Missing count column {} in {}'.format(s,csv_file))
        
    ## Create datetimes
    
    # print('Creating datetimes')
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
        
        date = row['Date']
        time = row['Time']
        datestring = date + ' ' + time
        dt = dateutil.parser.parse(datestring)
        assert dt.year >= 2015 and dt.year <= 2019
        df.loc[i_row,'datetime'] = dt
        
    # Make sure data are sorted chronologically
    #
    # In odd circumstances, they are not... so sort them first, but warn
    datetimes = list(df['datetime'])
    if not list_is_sorted(datetimes):
        print('Datetimes not sorted for {}'.format(csv_file))
    
    df = df.sort_values('datetime') 
    df.reset_index(drop=True, inplace=True)
    datetimes = list(df['datetime'])
    assert list_is_sorted(datetimes)

    # Debugging when I was trying to see what was up with the unsorted dates    
    if False:
        for i in range(0,len(datetimes)-1):
            dt = datetimes[i+1]
            prev_dt = datetimes[i]
            delta = dt - prev_dt
            assert delta >= datetime.timedelta(0)
    
    ## Parse into sequences    
    
    # print('Creating sequences')
    
    current_sequence_id = None
    next_frame_number = 0
    previous_datetime = None
        
    sequence_id_to_rows = defaultdict(list)
    
    # i_row = 0; row = df.iloc[i_row]    
    for i_row,row in df.iterrows():
        
        dt = row['datetime']
        assert dt is not None and isinstance(dt,datetime.datetime)
        
        # Start a new sequence if:
        #
        # * This image has no timestamp
        # * This image has a frame number of zero
        # * We have no previous image timestamp
        #
        if previous_datetime is None:
            delta = None
        else:
            delta = (dt - previous_datetime).total_seconds()
        
        # Start a new sequence if necessary
        if delta is None or delta > max_gap_within_sequence:
            next_frame_number = 0
            current_sequence_id = location_name + '_seq_' + str(dt) # str(uuid.uuid1())
            
        assert current_sequence_id is not None
        
        sequence_id_to_rows[current_sequence_id].append(i_row)
        df.loc[i_row,'seq_id'] = current_sequence_id
        df.loc[i_row,'synthetic_frame_number'] = next_frame_number
        next_frame_number = next_frame_number + 1
        previous_datetime = dt
        
    # ...for each row
    
    location_sequences = list(set(list(df['seq_id'])))
    location_sequences.sort()
    
    inconsistent_sequences = []
    
    
    ## Parse labels for each sequence
    
    # sequence_id = location_sequences[0]
    for sequence_id in location_sequences:
        
        sequence_row_indices = sequence_id_to_rows[sequence_id]
        assert len(sequence_row_indices) > 0
        
        # Row indices in a sequence should be adjacent
        if len(sequence_row_indices) > 1:
            d = np.diff(sequence_row_indices)
            assert(all(d==1))
        
        # sequence_df = df[df['seq_id']==sequence_id]
        sequence_df = df.iloc[sequence_row_indices]
        
        
        ## Determine what's present
        
        presence_columns_marked = []
        survey_species = []
        other_species = []
        
        # Be conservative; assume humans are present in all maintenance images
        opstates = set(sequence_df['opstate'])
        assert all([ ( (isinstance(s,float)) or (len(s.strip())== 0) or \
                      (s.strip() in valid_opstates)) for s in opstates]),\
            'Invalid optstate in: {}'.format(' | '.join(opstates))
        
        for presence_column in presence_columns:
                    
            presence_values = list(sequence_df[presence_column])
            
            # The presence columns are *almost* always identical for all images in a sequence        
            single_presence_value = (len(set(presence_values)) == 1)
            # assert single_presence_value
            if not single_presence_value:
                # print('Warning: presence value for {} is inconsistent for {}'.format(
                #   presence_column,sequence_id))
                inconsistent_sequences.append(sequence_id)                
            
            if any(presence_values):
                presence_columns_marked.append(presence_column)                
                
        # ...for each presence column
        
        # Tally up the standard (survey) species
        survey_species = [s.replace('present','') for s in presence_columns_marked if s != 'otherpresent']
        for opstate in opstates:
            if not isinstance(opstate,str):
                continue
            opstate = opstate.strip()
            if len(opstate) == 0:
                continue
            if opstate in opstate_mappings:
                    opstate = opstate_mappings[opstate]                
            if (opstate != 'normal') and (opstate not in survey_species):
                survey_species.append(opstate)
            
        # If no presence columns are marked, all counts should be zero
        if len(presence_columns_marked) == 0:
            
            # count_column = count_columns[0]
            for count_column in count_columns:
                
                values = list(set(list(sequence_df[count_column])))
                
                # Occasionally a count gets entered (correctly) without the presence column being marked
                # assert len(values) == 1 and values[0] == 0, 'Non-zero counts with no presence
                # columns marked for sequence {}'.format(sequence_id)
                if (not(len(values) == 1 and values[0] == 0)):
                    print('Warning: presence and counts are inconsistent for {}'.format(sequence_id))
                    
                    # Handle this by virtually checking the "right" box
                    for presence_column in presence_to_count_columns.keys():
                        count_columns_this_species = presence_to_count_columns[presence_column]
                        if count_column in count_columns_this_species:
                            if presence_column not in presence_columns_marked:
                                presence_columns_marked.append(presence_column)
                    
                    # Make sure we found a match
                    assert len(presence_columns_marked) > 0
                
        # Handle 'other' tags
        if 'otherpresent' in presence_columns_marked:
            
            sequence_otherwhats = set()
            sequence_comments = set()
            
            for i,r in sequence_df.iterrows():            
                otherwhat = r['otherwhat']
                if isinstance(otherwhat,str):
                    otherwhat = otherwhat.strip()
                    if len(otherwhat) > 0:
                        sequence_otherwhats.add(otherwhat)
                comment = r['comment']
                if isinstance(comment,str):
                    comment = comment.strip()
                    if len(comment) > 0:
                        sequence_comments.add(comment)
                
            freetext_species = []
            for s in sequence_otherwhats:
                freetext_species.append(s)
            for s in sequence_comments:
                freetext_species.append(s)
                
            counted_species = []
            
            otherpresent_columns = presence_to_count_columns['otherpresent']
            
            # column_name = otherpresent_columns[0]
            for column_name in otherpresent_columns:
            
                if column_name in sequence_df and column_name != 'other':
            
                    column_counts = list(sequence_df[column_name])
                    column_count_positive = any([c > 0 for c in column_counts])
                    
                    if column_count_positive:
                        # print('Found non-survey counted species column: {}'.format(column_name))
                        counted_species.append(column_name)
            
            # ...for each non-empty presence column
        
            # Very rarely, the "otherpresent" column is checked, but no more detail is available
            if not ( (len(freetext_species) > 0) or (len(counted_species) > 0) ):
                other_species.append('unknown')
                
            other_species += freetext_species
            other_species += counted_species
            
        # ...handling non-survey species
        
        all_species = other_species + survey_species
                 
        # Build the sequence data
        
        images = []
        # i_row = 0; row = sequence_df.iloc[i_row]
        for i_row,row in sequence_df.iterrows():
            im = {}
            # Only one folder used a single .csv file for two subfolders
            if ('RelativePath' in row) and (isinstance(row['RelativePath'],str)) \
                and (len(row['RelativePath'].strip()) > 0):
                assert 'IDFG-028' in location_name
                im['file_name'] = os.path.join(row['RelativePath'],row['File'])
            else:
                im['file_name'] = row['File']
            im['datetime'] = row['datetime']
            images.append(im)
            
        sequence = {}
        sequence['csv_source'] = csv_file
        sequence['sequence_id'] = sequence_id
        sequence['images'] = images
        sequence['species_present'] = all_species
        sequence['location'] = location_name
        
        sequences.append(sequence)
        
    # ...for each sequence

    return sequences

# ...def csv_to_sequences()


#%% Parse each .csv file into sequences (loop)

if __name__ == "__main__":
    
    #%%
    
    import multiprocessing
    multiprocessing.freeze_support()
    image_files,csv_files = get_files()
    
    #%%
    
    if n_threads == 1:
        
        # i_file = -1; csv_file = csv_files[i_file]
        sequences_by_file = []
        for i_file,csv_file in enumerate(csv_files):
            print('Processing file {} of {}'.format(i_file,len(csv_files)))
            sequences = csv_to_sequences(csv_file)
            sequences_by_file.append(sequences)
    
    else:
        
        pool = Pool(n_threads)
        sequences_by_file = list(pool.imap(csv_to_sequences,csv_files))

    
    #%% Save sequence data
        
    with open(sequence_info_cache,'w') as f:
        json.dump(sequences_by_file,f,indent=2,default=json_util.default)
        
    
    #%% Load sequence data
    
    if False:
        
        #%%
    
        with open(sequence_info_cache,'r') as f:
            sequences_by_file = json.load(f,object_hook=json_util.object_hook)
        
        
    #%% Validate file mapping (based on the existing enumeration)
    
    missing_images = []
    image_files_set = set(image_files)
    n_images_in_sequences = 0
    sequence_ids = set()
    
    # sequences = sequences_by_file[0]
    for i_sequences,sequences in enumerate(tqdm(sequences_by_file)):
        
        assert len(sequences) > 0
        csv_source = sequences[0]['csv_source']
        csv_file_absolute = os.path.join(input_base,csv_source)
        csv_folder = os.path.dirname(csv_file_absolute)
        assert os.path.isfile(csv_file_absolute)
        
        # sequence = sequences[0]
        for i_sequence,sequence in enumerate(sequences):
            
            assert sequence['csv_source'] == csv_source
            sequence_id = sequence['sequence_id']
            if sequence_id in sequence_ids:
                print('Warning: duplicate sequence for {}, creating new sequence'.format(sequence_id))
                sequence['sequence_id'] = sequence['sequence_id'] + '_' + str(i_sequences) + \
                    '_' + str(i_sequence)
                sequence_id = sequence['sequence_id']
                assert sequence_id not in sequence_ids
                
            sequence_ids.add(sequence_id)
            
            species_present = sequence['species_present']
            images = sequence['images']
            
            for im in images:
        
                n_images_in_sequences += 1
                image_file_relative = im['file_name']
                
                # Actually, one folder has relative paths
                # assert '\\' not in image_file_relative and '/' not in image_file_relative
                
                image_file_absolute = os.path.join(csv_folder,image_file_relative)
                image_file_container_relative = os.path.relpath(image_file_absolute,input_base)
                
                # os.startfile(csv_folder)
                # assert os.path.isfile(image_file_absolute)
                # found_file = os.path.isfile(image_file_absolute)
                found_file = image_file_container_relative in image_files_set
                if not found_file:
                    print('Warning: can\'t find image {}'.format(image_file_absolute))
                    missing_images.append(image_file_absolute)
                    
            # ...for each image
    
        # ...for each sequence
    
    # ...for each .csv file            
    
    print('{} of {} images missing ({} on disk)'.format(len(missing_images),n_images_in_sequences,
                                                        len(image_files)))
            
    
    #%% Load manual category mappings
    
    with open(category_mapping_file,'r') as f:
        category_mapping_lines = f.readlines()
        category_mapping_lines = [s.strip() for s in category_mapping_lines]

    category_mappings = {}
    for s in category_mapping_lines:
        tokens = s.split(',',1)
        category_name = tokens[0].strip()
        category_value = tokens[1].strip().replace('"','').replace(',','+')
        assert ',' not in category_name
        assert ',' not in category_value
        
        # The second column is blank when the first column already represents the category name
        if len(category_value) == 0:
            category_value = category_name
        category_mappings[category_name] = category_value
                
    
    #%% Convert to CCT .json (original strings)
            
    human_flagged_images = []
    with open(human_review_list,'r') as f:
        human_flagged_images = f.readlines()
        human_flagged_images = [s.strip().replace('/','\\') for s in human_flagged_images]
        human_flagged_images = set(human_flagged_images)
    print('Read {} human flagged images'.format(len(human_flagged_images)))
    
    annotations = []
    image_id_to_image = {}
    category_name_to_category = {}
    
    # Force the empty category to be ID 0
    empty_category_id = 0
    empty_category = {}
    empty_category['id'] = empty_category_id
    empty_category['name'] = 'empty'
    category_name_to_category['empty'] = empty_category
    
    human_category_id = 1
    human_category = {}
    human_category['id'] = human_category_id
    human_category['name'] = 'human'
    category_name_to_category['human'] = human_category
    
    next_category_id = 2
    
    annotation_ids = set()
    
    if False:
        target_folder = r'ClearCreek_mustelids\Summer2015\FS-035'
        for sequences in sequences_by_file:
            if target_folder in sequences[0]['csv_source']:
                break
            
    # For each .csv file...
    #
    # sequences = sequences_by_file[0]
    for sequences in tqdm(sequences_by_file):
            
        # For each sequence...
        #
        # sequence = sequences[0]
        for sequence in sequences:
        
            species_present = sequence['species_present']
            species_present = [s.lower().strip().replace(',',';') for s in species_present]
            
            sequence_images = sequence['images']
            location = sequence['location'].lower().strip()
            sequence_id = sequence['sequence_id']
            csv_source = sequence['csv_source']
            csv_folder_relative = os.path.dirname(csv_source)
            
            sequence_category_ids = set()
            
            # Find categories for this image                
            if len(species_present) == 0:
                
                sequence_category_ids.add(0)
                assert category_name_to_category['empty']['id'] == list(sequence_category_ids)[0]
                
            else:
                                
                # When 'unknown' is used in combination with another label, use that
                # label; the "unknown" here doesn't mean "another unknown species", it means
                # there is some other unknown property about the main species.
                if 'unknown' in species_present and len(species_present) > 1:
                    assert all([((s in category_mappings) or (s in valid_opstates) or \
                                 (s in opstate_mappings.values()))\
                                for s in species_present if s != 'unknown'])
                    species_present = [s for s in species_present if s != 'unknown']
                                        
                # category_name_string = species_present[0]
                for category_name_string in species_present:
                    
                    # This piece of text had a lot of complicated syntax in it, and it would have 
                    # been too complicated to handle in a general way
                    if 'coyotoes' in category_name_string:
                        # print('Ignoring category {}'.format(category_name_string))
                        continue
                    
                    if category_name_string not in category_mappings:
                        assert category_name_string in valid_opstates or \
                            category_name_string in opstate_mappings.values()                            
                    else:
                        category_name_string = category_mappings[category_name_string]
                    assert ',' not in category_name_string
                    
                    category_names = category_name_string.split('+')
                    assert len(category_names) <= 2
                    
                    # Don't process redundant labels
                    category_names = set(category_names)
                    
                    # category_name = category_names[0]
                    for category_name in category_names:
                    
                        if category_name == 'ignore':
                            continue
                    
                        category_name = category_name.replace('"','')                            
                                                                        
                        # If we've seen this category before...
                        if category_name in category_name_to_category:
                                
                            category = category_name_to_category[category_name]
                            category_id = category['id'] 
                          
                        # If this is a new category...
                        else:
                            
                            # print('Adding new category for {}'.format(category_name))
                            category_id = next_category_id
                            category = {}
                            category['id'] = category_id
                            category['name'] = category_name
                            category_name_to_category[category_name] = category
                            next_category_id += 1
                            
                        sequence_category_ids.add(category_id)
                        
                    # ...for each category (inner)
                    
                # ...for each category (outer)
                    
            # ...if we do/don't have species in this sequence
            
            # We should have at least one category assigned (which may be "empty" or "unknown")
            assert len(sequence_category_ids) > 0
               
            # assert len(sequence_category_ids) > 0
                        
            # Was any image in this sequence manually flagged as human?
            for i_image,im in enumerate(sequence_images):
                
                file_name_relative = os.path.join(csv_folder_relative,im['file_name'])
                if file_name_relative in human_flagged_images:
                    # print('Flagging sequence {} as human based on manual review'.format(sequence_id))
                    assert human_category_id not in sequence_category_ids
                    sequence_category_ids.add(human_category_id)
                    break   
                    
            # For each image in this sequence...
            # 
            # i_image = 0; im = images[i_image]
            for i_image,im in enumerate(sequence_images):
                
                image_id = sequence_id + '_' + im['file_name']
                assert image_id not in image_id_to_image
                
                output_im = {}
                output_im['id'] = image_id
                output_im['file_name'] = os.path.join(csv_folder_relative,im['file_name'])
                output_im['seq_id'] = sequence_id
                output_im['seq_num_frames'] = len(sequence)
                output_im['frame_num'] = i_image
                output_im['datetime'] = str(im['datetime'])
                output_im['location'] = location
                
                image_id_to_image[image_id] = output_im
                
                # Create annotations for this image
                for i_ann,category_id in enumerate(sequence_category_ids):
                    
                    ann = {}
                    ann['id'] = 'ann_' + image_id + '_' + str(i_ann)
                    assert ann['id'] not in annotation_ids
                    annotation_ids.add(ann['id'])
                    ann['image_id'] = image_id
                    ann['category_id'] = category_id
                    ann['sequence_level_annotation'] = True
                    annotations.append(ann)

            # ...for each image in this sequence

        # ...for each sequence
                    
    # ...for each .csv file
    
    images = list(image_id_to_image.values())
    categories = list(category_name_to_category.values())
    print('Loaded {} annotations in {} categories for {} images'.format(
        len(annotations),len(categories),len(images)))
    
    # Verify that all images have annotations
    image_id_to_annotations = defaultdict(list)
    
    # ann = ict_data['annotations'][0]
    
    # For debugging only
    categories_to_counts = defaultdict(int)
    for ann in tqdm(annotations):
        image_id_to_annotations[ann['image_id']].append(ann)
        categories_to_counts[ann['category_id']] = categories_to_counts[ann['category_id']] + 1
        
    for im in tqdm(images):        
        image_annotations = image_id_to_annotations[im['id']]
        assert len(image_annotations) > 0
        
    
    #%% Create output (original strings)
    
    info = {}
    info['contributor'] = 'Idaho Department of Fish and Game'
    info['description'] = 'Idaho Camera traps'    
    info['version'] = '2021.07.19'
    
    output_data = {}
    output_data['images'] = images
    output_data['annotations'] = annotations
    output_data['categories'] = categories
    output_data['info'] = info
            
    with open(output_json_original_strings,'w') as f:
        json.dump(output_data,f,indent=1)
        
    
    #%% Validate .json file

    from data_management.databases import sanity_check_json_db

    options = sanity_check_json_db.SanityCheckOptions()
    options.baseDir = input_base
    options.bCheckImageSizes = False
    options.bCheckImageExistence = False
    options.bFindUnusedImages = False

    _, _, _ = sanity_check_json_db.sanity_check_json_db(output_json_original_strings, options)
    
    
    #%% Preview labels
            
    from visualization import visualize_db
    
    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = 1000
    viz_options.trim_to_images_with_bboxes = False
    viz_options.add_search_links = False
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    viz_options.include_filename_links = True
    
    viz_options.classes_to_exclude = ['empty','deer','elk']
    html_output_file, _ = visualize_db.process_images(db_path=output_json_original_strings,
                                                             output_dir=os.path.join(
                                                             output_base,'preview'),
                                                             image_base_dir=input_base,
                                                             options=viz_options)
    os.startfile(html_output_file)
    
    
    #%% Look for humans that were found by MegaDetector that haven't already been identified as human
    
    # This whole step only needed to get run once
    
    if False:
        
        pass
    
        #%%
        
        human_confidence_threshold = 0.5
        
        # Load MD results
        with open(megadetector_results_file,'r') as f:
            md_results = json.load(f)
            
        # Get a list of filenames that MD tagged as human
        
        human_md_categories =\
            [category_id for category_id in md_results['detection_categories'] if \
             ((md_results['detection_categories'][category_id] == 'person') or \
              (md_results['detection_categories'][category_id] == 'vehicle'))]
        assert len(human_md_categories) == 2
            
        # im = md_results['images'][0]
        md_human_images = set()
        
        for im in md_results['images']:
            if 'detections' not in im:
                continue
            if im['max_detection_conf'] < human_confidence_threshold:
                  continue
            for detection in im['detections']:
                if detection['category'] not in human_md_categories:
                    continue
                elif detection['conf'] < human_confidence_threshold:
                    continue
                else:
                    md_human_images.add(im['file'])
                    break
            
            # ...for each detection
            
        # ...for each image
        
        print('MD found {} potential human images (of {})'.format(
            len(md_human_images),len(md_results['images'])))    
    
        # Map images to annotations in ICT
        
        with open(output_json_original_strings,'r') as f:
            ict_data = json.load(f)
        
        category_id_to_name = {c['id']:c['name'] for c in categories}
        
        image_id_to_annotations = defaultdict(list)
        
        # ann = ict_data['annotations'][0]
        for ann in tqdm(ict_data['annotations']):
            image_id_to_annotations[ann['image_id']].append(ann)
            
        human_ict_categories = ['human']
        manual_human_images = set()
        
        # For every image
        # im = ict_data['images'][0]
        for im in tqdm(ict_data['images']):
                
            # Does this image already have a human annotation?
            manual_human = False
            
            annotations = image_id_to_annotations[im['id']]
            assert len(annotations) > 0
            
            for ann in annotations:
                category_name = category_id_to_name[ann['category_id']]
                if category_name in human_ict_categories:        
                    manual_human_images.add(im['file_name'].replace('\\','/'))
            
            # ...for each annotation
            
        # ...for each image
        
        print('{} images identified as human in source metadata'.format(len(manual_human_images)))
            
        missing_human_images = []
        
        for fn in md_human_images:
            if fn not in manual_human_images:
                missing_human_images.append(fn)
        
        print('{} potentially untagged human images'.format(len(missing_human_images)))
    
    
        #%% Copy images for review to a new folder
                
        os.makedirs(human_review_folder,exist_ok=True)
        missing_human_images.sort()
        
        # fn = missing_human_images[0]
        for i_image,fn in enumerate(tqdm(missing_human_images)):
            input_fn_absolute = os.path.join(input_base,fn).replace('\\','/')
            assert os.path.isfile(input_fn_absolute)
            output_path = os.path.join(human_review_folder,str(i_image).zfill(4) + '_' + fn.replace('/','~'))
            shutil.copyfile(input_fn_absolute,output_path)
        
        
        #%% Manual step...
        
        # Copy any images from that list that have humans in them to...
        human_review_selection_folder = r'H:\idaho-camera-traps\human_review_selections'
        assert os.path.isdir(human_review_selection_folder)
        
        
        #%% Create a list of the images we just manually flagged
        
        human_tagged_filenames = os.listdir(human_review_selection_folder)
        human_tagged_relative_paths = []
        # fn = human_tagged_filenames[0]
        for fn in human_tagged_filenames:
            
            # E.g. '0000_Beaverhead_elk~AM174~Trip 1~100RECNX~IMG_1397.JPG'
            relative_path = fn[5:].replace('~','/')
            human_tagged_relative_paths.append(relative_path)
        
        with open(human_review_list,'w') as f:
            for s in human_tagged_relative_paths:
                f.write(s + '\n')
            

    #%% Translate location, image, sequence IDs
    
    # Load mappings if available        
    if (not force_generate_mappings) and (os.path.isfile(id_mapping_file)):
        
        print('Loading ID mappings from {}'.format(id_mapping_file))
        
        with open(id_mapping_file,'r') as f:
            mappings = json.load(f)
    
        image_id_mappings = mappings['image_id_mappings']
        annotation_id_mappings = mappings['annotation_id_mappings']
        location_id_mappings = mappings['location_id_mappings']
        sequence_id_mappings = mappings['sequence_id_mappings']
        
    else:
        
        # Generate mappings
        mappings = {}
        
        next_location_id = 0
        location_id_string_to_n_sequences = defaultdict(int)
        location_id_string_to_n_images = defaultdict(int)
        
        image_id_mappings = {}
        annotation_id_mappings = {}
        location_id_mappings = {}
        sequence_id_mappings = {}
        
        for im in tqdm(images):
                            
            # If we've seen this location before...
            if im['location'] in location_id_mappings:
                location_id = location_id_mappings[im['location']]
            else:
                # Otherwise assign a string-formatted int as the ID
                location_id = str(next_location_id)
                
                location_id_mappings[im['location']] = location_id
                next_location_id += 1
            
            # If we've seen this sequence before...
            if im['seq_id'] in sequence_id_mappings:
                sequence_id = sequence_id_mappings[im['seq_id']]
            else:
                # Otherwise assign a string-formatted int as the ID
                n_sequences_this_location = location_id_string_to_n_sequences[location_id]
                sequence_id = 'loc_{}_seq_{}'.format(
                    location_id.zfill(4),str(n_sequences_this_location).zfill(6))
                sequence_id_mappings[im['seq_id']] = sequence_id
                
                n_sequences_this_location += 1
                location_id_string_to_n_sequences[location_id] = n_sequences_this_location
                
            assert im['id'] not in image_id_mappings
            
            # Assign an image ID
            
            n_images_this_location = location_id_string_to_n_images[location_id]                
            image_id_mappings[im['id']] = 'loc_{}_im_{}'.format(
                location_id.zfill(4),str(n_images_this_location).zfill(6))
            
            n_images_this_location += 1
            location_id_string_to_n_images[location_id] = n_images_this_location
        
        # ...for each image
        
        # Assign annotation mappings
        for i_ann,ann in enumerate(tqdm(annotations)):
            assert ann['image_id'] in image_id_mappings
            assert ann['id'] not in annotation_id_mappings
            annotation_id_mappings[ann['id']] = 'ann_{}'.format(str(i_ann).zfill(8))
            
        mappings['image_id_mappings'] = image_id_mappings
        mappings['annotation_id_mappings'] = annotation_id_mappings
        mappings['location_id_mappings'] = location_id_mappings
        mappings['sequence_id_mappings'] = sequence_id_mappings
                    
        # Save mappings
        with open(id_mapping_file,'w') as f:
            json.dump(mappings,f,indent=2)
            
        print('Saved ID mappings to {}'.format(id_mapping_file))
       
        # Back this file up, lest we should accidentally re-run this script
        # with force_generate_mappings = True and overwrite the mappings we used.
        datestr = str(datetime.datetime.now()).replace(':','-')
        backup_file = id_mapping_file.replace('.json','_' + datestr + '.json')
        shutil.copyfile(id_mapping_file,backup_file)
        
    # ...if we are/aren't re-generating mappings
    
    
    #%% Apply mappings
    
    for im in images:
        im['id'] = image_id_mappings[im['id']]
        im['seq_id'] = sequence_id_mappings[im['seq_id']]
        im['location'] = location_id_mappings[im['location']]
    for ann in annotations:
        ann['id'] = annotation_id_mappings[ann['id']]
        ann['image_id'] = image_id_mappings[ann['image_id']]
    
    print('Applied mappings')
    
    
    #%% Write new dictionaries (modified strings, original files)
    
    output_data = {}
    output_data['images'] = images
    output_data['annotations'] = annotations
    output_data['categories'] = categories
    output_data['info'] = info
    
    with open(output_json_remapped_ids,'w') as f:
        json.dump(output_data,f,indent=2)
        
    
    #%% Validate .json file (modified strings, original files)

    from data_management.databases import sanity_check_json_db

    options = sanity_check_json_db.SanityCheckOptions()
    options.baseDir = input_base
    options.bCheckImageSizes = False
    options.bCheckImageExistence = False
    options.bFindUnusedImages = False

    _, _, _ = sanity_check_json_db.sanity_check_json_db(output_json_remapped_ids, options)
    
    
    #%% Preview labels (original files)
            
    from visualization import visualize_db
    
    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = 1000
    viz_options.trim_to_images_with_bboxes = False
    viz_options.add_search_links = False
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    viz_options.include_filename_links = True
    
    # viz_options.classes_to_exclude = ['empty','deer','elk']
    # viz_options.classes_to_include = ['bobcat']
    viz_options.classes_to_include = [viz_options.multiple_categories_tag] 
    
    html_output_file, _ = visualize_db.process_images(db_path=output_json_remapped_ids,
                                                             output_dir=os.path.join(
                                                             output_base,'preview'),
                                                             image_base_dir=input_base,
                                                             options=viz_options)
    os.startfile(html_output_file)


    #%% Copy images to final output folder (prep)
    
    force_copy = False
    
    with open(output_json_remapped_ids,'r') as f:
        d = json.load(f)
        
    images = d['images']
    
    private_categories = ['human','domestic dog','vehicle']
    
    private_image_ids = set()

    category_id_to_name = {c['id']:c['name'] for c in d['categories']}
    
    # ann = d['annotations'][0]
    for ann in d['annotations']:
        category_name = category_id_to_name[ann['category_id']]
        if category_name in private_categories:
            private_image_ids.add(ann['image_id'])
    
    print('Moving {} of {} images to the private folder'.format(len(private_image_ids),len(images)))
    
    def process_image(im):
        
        input_relative_path = im['file_name']
        input_absolute_path = os.path.join(input_base,input_relative_path)
        
        if not os.path.isfile(input_absolute_path):
            print('Warning: file {} is not available'.format(input_absolute_path))
            return
        
        location = im['location']
        image_id = im['id']
        
        location_folder = 'loc_' + location.zfill(4)
        assert location_folder in image_id
        
        output_relative_path =  location_folder + '/' + image_id + '.jpg'
        
        # Is this a public or private image?
        private_image = (image_id in private_image_ids)
                
        # Generate absolute path
        if private_image:
            output_absolute_path = os.path.join(output_image_base_private,output_relative_path)
        else:
            output_absolute_path = os.path.join(output_image_base_public,output_relative_path)
                
        # Copy to output
        output_dir = os.path.dirname(output_absolute_path)
        os.makedirs(output_dir,exist_ok=True)
        
        if force_copy or (not os.path.isfile(output_absolute_path)):
            shutil.copyfile(input_absolute_path,output_absolute_path)
    
        # Update the filename reference
        im['file_name'] = output_relative_path

    # ...def process_image(im)
    
    
    #%% Copy images to final output folder (execution)
        
    # For each image
    if n_threads_file_copy == 1:
        # im = images[0]
        for im in tqdm(images):    
            process_image(im)
    else:
        pool = ThreadPool(n_threads_file_copy)
        pool.map(process_image,images)

    print('Finished copying, writing .json output')
    
    # Write output .json
    with open(output_json,'w') as f:
        json.dump(d,f,indent=1)
    
    
    #%% Make sure the right number of images got there
    
    from pathlib import Path
    all_output_files = []
    all_output_files_list = os.path.join(output_base,'all_output_files.json')
    
    for path in Path(output_image_base).rglob('*.*'):
        path = str(path)
        path = os.path.relpath(path,output_image_base)
        all_output_files.append(path)
    with open(all_output_files_list,'w') as f:
        json.dump(all_output_files,f,indent=1)

    print('Enumerated {} output files (of {} images)'.format(len(all_output_files),len(images)))


    #%% Validate .json file (final filenames)

    from data_management.databases import sanity_check_json_db

    options = sanity_check_json_db.SanityCheckOptions()
    options.baseDir = input_base
    options.bCheckImageSizes = False
    options.bCheckImageExistence = False
    options.bFindUnusedImages = False

    _, _, _ = sanity_check_json_db.sanity_check_json_db(output_json, options)
    
    
    #%% Preview labels (final filenames)
    
    from visualization import visualize_db
    
    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = 1500
    viz_options.trim_to_images_with_bboxes = False
    viz_options.add_search_links = False
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    viz_options.include_filename_links = True
    
    # viz_options.classes_to_exclude = ['empty','deer','elk']
    viz_options.classes_to_include = ['bear','mountain lion']
    # viz_options.classes_to_include = ['horse']
    # viz_options.classes_to_include = [viz_options.multiple_categories_tag] 
    # viz_options.classes_to_include = ['human','vehicle','domestic dog'] 
    
    html_output_file, _ = visualize_db.process_images(db_path=output_json,
                                                             output_dir=os.path.join(
                                                             output_base,'final-preview-01'),
                                                             image_base_dir=output_image_base_public,
                                                             options=viz_options)
    os.startfile(html_output_file)


    #%% Create zipfiles
    
    #%% List public files
    
    from pathlib import Path
    all_public_output_files = []
    all_public_output_files_list = os.path.join(output_base,'all_public_output_files.json')
    
    if not os.path.isfile(all_public_output_files_list):
        for path in Path(output_image_base_public).rglob('*.*'):
            path = str(path)
            path = os.path.relpath(path,output_image_base)
            all_public_output_files.append(path)
        with open(all_public_output_files_list,'w') as f:
            json.dump(all_public_output_files,f,indent=1)
    else:
        with open(all_public_output_files_list,'r') as f:
            all_public_output_files = json.load(f)
            
    print('Enumerated {} public output files'.format(len(all_public_output_files)))


    #%% Find the size of each file
    
    filename_to_size = {}
    
    all_public_output_sizes_list = os.path.join(output_base,'all_public_output_sizes.json')
    
    if not os.path.isfile(all_public_output_sizes_list):
        # fn = all_public_output_files[0]
        for fn in tqdm(all_public_output_files):
            p = os.path.join(output_image_base,fn)
            assert os.path.isfile(p)
            filename_to_size[fn] = os.path.getsize(p)
            
        with open(all_public_output_sizes_list,'w') as f:
            json.dump(filename_to_size,f,indent=1)
    else:
        with open(all_public_output_sizes_list,'r') as f:
            filename_to_size = json.load(f)
                
    assert len(filename_to_size) == len(all_public_output_files)
    
    
    #%% Split into chunks of approximately-equal size
    
    import humanfriendly
    total_size = sum(filename_to_size.values())
    print('{} in {} files'.format(humanfriendly.format_size(total_size),len(all_public_output_files)))
    
    bytes_per_part = 320e9
    
    file_lists = []
        
    current_file_list = []
    n_bytes_current_file_list = 0
    
    for fn in all_public_output_files:
        size = filename_to_size[fn]
        current_file_list.append(fn)
        n_bytes_current_file_list += size
        if n_bytes_current_file_list > bytes_per_part:
            file_lists.append(current_file_list)
            current_file_list = []
            n_bytes_current_file_list = 0
    # ...for each file
    
    file_lists.append(current_file_list)
            
    assert sum([len(l) for l in file_lists]) == len(all_public_output_files)
        
    print('List sizes:')
    for l in file_lists:
        print(len(l))
    
    
    #%% Create a zipfile for each chunk
    
    from zipfile import ZipFile
    import zipfile
    import os
    
    def create_zipfile(i_file_list):
        
        file_list = file_lists[i_file_list]
        zipfile_name = os.path.join('k:\\idaho-camera-traps-images.part_{}.zip'.format(i_file_list))
        
        print('Processing archive {} to file {}'.format(i_file_list,zipfile_name))
        
        with ZipFile(zipfile_name, 'w') as zipObj:
            
            for filename_relative in file_list:
           
                assert filename_relative.startswith('public')
                filename_absolute = os.path.join(output_image_base,filename_relative)
                zipObj.write(filename_absolute.replace('\\','/'), 
                             filename_relative, compress_type=zipfile.ZIP_STORED)
                
            # ...for each filename
            
        # with ZipFile()
        
    # ...def create_zipfile()
    
    # i_file_list = 0; file_list = file_lists[i_file_list]
    
    n_zip_threads = 1 # len(file_lists)
    if n_zip_threads == 1:
        for i_file_list in range(0,len(file_lists)):
            create_zipfile(i_file_list)
    else:
        pool = ThreadPool(n_zip_threads)
        indices = list(range(0,len(file_lists)))
        pool.map(create_zipfile,indices)

# ....if __name__ == "__main__"