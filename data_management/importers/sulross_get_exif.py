import os
import json
from tqdm import tqdm

import exiftool


image_ids_path = '/home/beaver/cameratraps/data/sulross/20190522_image_ids.json'

data_dir = '/home/beaver/cameratraps/mnt/sulross'


def get_metadata():
    image_ids = json.load(open(image_ids_path))

    image_id_to_metadata = {}

    batch_size = 20  # exiftool can process a batch of images at a time, but bottleneck is blobfuse reading the images

    num_images_processed = 0

    with exiftool.ExifTool() as et:
        for i in tqdm(range(0, len(image_ids), batch_size)):
            batch_ids = image_ids[i: i + batch_size]

            batch_paths = [os.path.join(data_dir, i) for i in batch_ids]

            try:
                metadatas = et.get_metadata_batch(batch_paths)

                for id, metadata in zip(batch_ids, metadatas):
                    image_id_to_metadata[id] = metadata['XMP:HierarchicalSubject']
            except Exception as e:
                print('Exception! {}'.format(e))
                continue

            num_images_processed += batch_size
            if num_images_processed % 1000 == 0:
                print('Finished processing {} images; image ID {}'.format(
                    num_images_processed, image_ids[num_images_processed - 1]))
                print(image_id_to_metadata[id])
                print()

            # checkpoint
            if num_images_processed % 10000 == 0:
                print('Saving results so far...')
                with open('/home/beaver/cameratraps/data/sulross/20190522_metadata.json', 'w') as f:
                    json.dump(image_id_to_metadata, f, indent=1)

    print('Length of meta data read: ', len(image_id_to_metadata))
    with open('/home/beaver/cameratraps/data/sulross/20190522_metadata.json', 'w') as f:
        json.dump(image_id_to_metadata, f, indent=1)
    print('Results saved. Done!')


if __name__ == '__main__':
    get_metadata()
