#
# idfg_iwildcam_lila_prep.py
#
# Adding class labels (from the private test .csv) to the iWildCam 2019 IDFG 
# test set, in preparation for release on LILA.
#
# This version works from the iWildCam source images.
#

#%% Imports and constants

import json
import os
import numpy as np
import dateutil 
import pandas as pd
import datetime
 
from tqdm import tqdm
from bson import json_util

from collections import defaultdict

# Multi-threading for .csv file comparison and image existence validation
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.pool import Pool as Pool
n_threads = 14

input_base = r'i:\idfg-images'
output_base = r'h:\idaho-camera-traps'
category_mapping_file = os.path.join(output_base,'category_mapping.csv')

output_json_original_strings = os.path.join(output_base,'idaho-camera-traps-tmp.json')

# List of images used in the iWildCam competition
iwildcam_image_list = r"h:\idaho-camera-traps\v0\iWildCam_IDFG_ml.json"
sequence_info_cache = os.path.join(output_base,'sequence_info.json')

assert os.path.isfile(iwildcam_image_list)
assert os.path.isdir(input_base)
assert os.path.isdir(output_base)

valid_opstates = ['normal','maintenance','snow on lens','foggy lens','foggy weather',
                  'malfunction','misdirected','snow on lense','poop/slobber','sun','tilted','vegetation obstruction']
opstate_mappings = {'snow on lense':'snow on lens','poop/slobber':'lens obscured','maintenance':'human'}
                
survey_species_presence_columns = ['elkpresent','deerpresent','prongpresent']

presence_to_count_columns = {
    'otherpresent':['MooseAntlerless','MooseCalf','MooseOther','MooseBull','MooseUnkn','BlackBearAdult','BlackBearCub','LionAdult',
                    'LionKitten','WolfAdult','WolfPup','CattleCow','CattleCalf','other'],
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

    location_name = '_'.join(csv_file.split('\\')[0:-1]).replace(' ','')
    
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
    
    max_gap_within_sequence = 10
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
        assert all([ ( (isinstance(s,float)) or (len(s.strip())== 0) or (s.strip() in valid_opstates)) for s in opstates]),\
            'Invalid optstate in: {}'.format(' | '.join(opstates))
        
        for presence_column in presence_columns:
                    
            presence_values = list(sequence_df[presence_column])
            
            # The presence columns are *almost* always identical for all images in a sequence        
            single_presence_value = (len(set(presence_values)) == 1)
            # assert single_presence_value
            if not single_presence_value:
                # print('Warning: presence value for {} is inconsistent for {}'.format(presence_column,sequence_id))
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
                # assert len(values) == 1 and values[0] == 0, 'Non-zero counts with no presence columns marked for sequence {}'.format(sequence_id)
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
            if ('RelativePath' in row) and (isinstance(row['RelativePath'],str)) and (len(row['RelativePath'].strip()) > 0):
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
        
    #%%
    
    if False:
        
        pass
    
        #%% Save sequence data
        
        with open(sequence_info_cache,'w') as f:
            json.dump(sequences_by_file,f,indent=2,default=json_util.default)
            
        
        #%% Validate file mapping (based on the existing enumeration)
        
        missing_images = []
        image_files_set = set(image_files)
        n_images_in_sequences = 0
        sequence_ids = set()
        
        # sequences = sequences_by_file[0]
        for sequences in tqdm(sequences_by_file):
            
            assert len(sequences) > 0
            csv_source = sequences[0]['csv_source']
            csv_file_absolute = os.path.join(input_base,csv_source)
            csv_folder = os.path.dirname(csv_file_absolute)
            assert os.path.isfile(csv_file_absolute)
            
            # sequence = sequences[0]
            for sequence in sequences:
                
                assert sequence['csv_source'] == csv_source
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
                
        annotations = []
        image_id_to_image = {}
        category_name_to_category = {}
        warned_categories = set()
        
        # Force the empty category to be ID 0
        empty_category = {}
        empty_category['id'] = 0
        empty_category['name'] = 'empty'
        category_name_to_category['empty'] = empty_category
        next_category_id = 1
        
        annotation_ids = set()
        
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
                    
                    sequence_category_ids = [0]
                    assert category_name_to_category['empty']['id'] == sequence_category_ids[0]
                    
                else:
                    
                    # When 'unknown' is used in combination with another label, use that
                    # label; the "unknown" here doesn't mean "another unknown species", it means
                    # there is some other unknown property about the main species.
                    if 'unknown' in species_present and len(species_present) > 1:
                        assert all([((s in category_mappings) or (s in valid_opstates) or (s in opstate_mappings.values()))\
                                    for s in species_present if s != 'unknown'])
                        species_present = [s for s in species_present if s != 'unknown']
                    
                    for category_name_string in species_present:
                        
                        # This piece of text had a lot of complicated syntax in it, and it would have 
                        # been too complicated to handle in a general way
                        if 'coyotoes' in category_name_string:
                            print('Ignoring category {}'.format(category_name_string))
                            continue
                        
                        if category_name_string not in category_mappings:
                            if category_name_string not in warned_categories:
                                print('Warning: category {} not in mappings'.format(category_name_string))
                                warned_categories.add(category_name_string)
                        else:
                            category_name_string = category_mappings[category_name_string]
                        assert ',' not in category_name_string
                        
                        category_names = category_name_string.split('+')
                        assert len(category_names) <= 2
                        
                        # Don't process redundant labels
                        category_names = set(category_names)
                        
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
                                
                                print('Adding new category for {}'.format(category_name))
                                category_id = next_category_id
                                category = {}
                                category['id'] = category_id
                                category['name'] = category_name
                                category_name_to_category[category_name] = category
                                next_category_id += 1
                                
                            sequence_category_ids.add(category_id)
                            
                        # ...if we have/haven't seen this species before
                        
                    # ...for each species present
                    
                # ...if we do/don't have species in this image
                
                # For each image...
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
        
        if False:
                        
            with open(category_mapping_file,'w') as f:
                for c in categories:
                    f.write(c['name'].replace(',',';') + ',\n')
                    

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
        