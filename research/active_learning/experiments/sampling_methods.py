import argparse, random, sys, time
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sys.path.append("..")
from DL.utils import *
from DL.networks import *
from Database.DB_models import *
from DL.sqlite_data_loader import SQLDataLoader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', default='missouricameratraps', type=str, help='Name of the training (target) data Postgres DB.')
    parser.add_argument('--db_user', default='user', type=str, help='Name of the user accessing the Postgres DB.')
    parser.add_argument('--db_password', default='password', type=str, help='Password of the user accessing the Postgres DB.')
    parser.add_argument('--num', default=5000, type=int, help='Number of samples to draw from dataset to get embedding features.')
    parser.add_argument('--crop_dir', type=str, help='Path to directory with cropped images to get embedding features for.')
    parser.add_argument('--base_model', type=str, help='Path to latest embedding model checkpoint.')
    parser.add_argument('--random_seed', default=1234, type=int, help='Random seed to get same samples from database.')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    BASE_MODEL = args.base_model
    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password

    # Connect to database and sample a dataset
    target_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
    target_db.connect(reuse_if_open=True)
    db_proxy.initialize(target_db)
    dataset_query = Detection.select(Detection.image_id, Oracle.label, Detection.kind).join(Oracle).limit(args.num)
    dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8, limit=args.num)
    imagepaths = dataset.getallpaths()

    # Load the saved embedding model from the checkpoint
    checkpoint = load_checkpoint(BASE_MODEL)
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # Update the dataset embedding
    dataset.updateEmbedding(model)
    
    # Get a random query image
    query_idx = np.random.randint(len(dataset.samples))
    query_img = dataset.loader(imagepaths[query_idx].split('.')[0])
    query_img.save("query_img.png")
    
    
    # # # # IMAGES IN THE SAME SEQUENCE # # # #
    matching_image_entries = (Image
                            .select(Image.seq_id, Image.seq_num_frames, Image.frame_num)
                            .where((Image.file_name == imagepaths[query_idx])))
    mie = matching_image_entries.get()
    if mie.seq_num_frames > 1:
        images_in_seq = (Image
                        .select(Image.file_name)
                        .where((Image.seq_id == mie.seq_id) & (Image.file_name << imagepaths))
                        )
    images_in_seq = sorted(list(set([i.file_name for i in images_in_seq])))
    seq_img_idx = [imagepaths.index(im) for im in images_in_seq]
    for i in range(len(seq_img_idx)):
        if images_in_seq[i] != imagepaths[query_idx]:
            img = dataset.loader(images_in_seq[i].split('.')[0])
            img.save('same_seq_img%d.png'%i)


    # assert 2==3, 'break'

    # # # # CLOSEST IN (EMBEDDING) FEATURE SPACE # # # #
    timer = time.time()
    nbrs = NearestNeighbors(n_neighbors=11).fit(dataset.em)
    print('Finished fitting nearest neighbors for whole dataset in %0.2f seconds'%(float(time.time() - timer)))
    distances, indices = nbrs.kneighbors(dataset.em)
    query_nbrs_indices = indices[query_idx, 1:11]
    for i in range(len(query_nbrs_indices)):
        nbr_idx = query_nbrs_indices[i]
        nbr_img = dataset.loader(imagepaths[nbr_idx].split('.')[0])
        nbr_img.save("embedding_nnbr_img%d.png"%i)


    print('Success!')


if __name__ == '__main__':
    main()