#
# Test script for pushing annotations to the eMammal db
#

#%% Imports

import sys
import json
import argparse
import pymysql
import config as cfg

from tqdm import tqdm
from enum import Enum


#%% Database functions

class Categories(Enum):
    animal = 1
    person = 2
    vehicle = 3

mysql_connection = pymysql.connect( host=cfg.host, 
                                user=cfg.username, 
                                passwd=cfg.password, 
                                db=cfg.database, 
                                port=cfg.port)

def update_data(sql):
    with mysql_connection.cursor() as cursor:
        cursor.execute(sql)
        
def get_records_all(sql):
    with mysql_connection.cursor() as cursor:
        sql = sql
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows

def format_data_print_deployments(rows):
    count = 0
    result = []
    for row in rows:
            count += 1
            print("{}. {}-{}".format(str(count), row[0],row[1]))
            result.append((count, row[0], row[1]))

    return result


#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input .json filename')
   
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args() 

    print("Enter the number of the deployment:")
    
    rows = get_records_all(''' select * from deployment ''')
    deployments = format_data_print_deployments(rows)
    print("\n")
    deployment_choice = input()
    deployment_id = deployments[int(deployment_choice)][1]
   
    print(deployment_id)

    # TODO: check project ID ?
    sql = ''' SELECT emammal_project_taxa_id FROM wild_id.emammal_project_taxa
            where species in ("No Animal", "Unknown Animal", "Homo sapiens", "Vehicle") '''


    emammal_categories = get_records_all(sql)

    with open(args.input_file) as f:
        data = json.load(f)

    images = data['images']
    emammal_category = 0
    for index, im in tqdm(enumerate(images), total=len(images)):      
        fn = im['file']

        if len(im['detections']) <= 0:
            image_type_id = 2
        
            # No-animal category
            emammal_categories = emammal_categories[0]
        else:            
            max_conf = im['max_detection_conf']
            detection = [k for k in im['detections'] if k['conf'] == max_conf]
            category= int(detection[0]['category'])

            if category == Categories.animal:
                image_type_id = 1
                emammal_category = emammal_categories[1]
            else:
                image_type_id = 5
                if category == Categories.person:
                    emammal_category = emammal_categories[2]
                elif category == Categories.vehicle:
                    emammal_category = emammal_categories[3]

            sql = """ UPDATE wild_id.emammal_sequence_annotation,
                      wild_id.image,
                      wild_id.image_sequence,
                      wild_id.deployment
                      SET wild_id.emammal_sequence_annotation.project_taxa_id = 4
                      WHERE wild_id.image.image_sequence_id = wild_id.emammal_sequence_annotation.sequence_id
                      AND wild_id.image_sequence.deployment_id = wild_id.deployment.deployment_id
                      AND wild_id.image.raw_name = '{}' """.format(fn)

            
            print(sql)
            update_data(sql)
            mysql_connection.commit()


if __name__ == '__main__':
    main()

