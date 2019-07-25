import argparse, random, sys, time
import PIL
import torch
from torchvision.transforms import *

sys.path.append("..")
from DL.utils import *
from DL.networks import *
from Database.DB_models import *
from DL.sqlite_data_loader import SQLDataLoader

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True).cuda()
    def close(self):
        self.hook.remove()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', default='missouricameratraps', type=str, help='Name of the training (target) data Postgres DB.')
    parser.add_argument('--db_user', default='user', type=str, help='Name of the user accessing the Postgres DB.')
    parser.add_argument('--db_password', default='password', type=str, help='Password of the user accessing the Postgres DB.')
    parser.add_argument('--num', default=1000, type=int, help='Number of samples to draw from dataset to get embedding features.')
    parser.add_argument('--crop_dir', type=str, help='Path to directory with cropped images to get embedding features for.')
    parser.add_argument('--base_model', type=str, help='Path to latest embedding model checkpoint.')
    parser.add_argument('--random_seed', default=1234, type=int, help='Random seed to get same samples from database.')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    BASE_MODEL = args.base_model

    # Load the saved embedding model from the checkpoint
    checkpoint = load_checkpoint(BASE_MODEL)
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # model_inner_resnet = list(model.children())[0].inner_model

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

    sample_image_path = '0ef0f79f-7b58-473d-abbf-75bba59e834d'
    dataset.image_mode()
    sample_image = dataset.loader(sample_image_path)
    print(sample_image.size)
    sample_image = dataset.eval_transform(sample_image)
    print(sample_image.shape)
    

    # sample_image = PILImage.open(sample_image_path).convert('RGB')
    # sample_image = transforms.Compose([Resize([256, 256]), CenterCrop(([[224,224]])), ToTensor(), Normalize([0.369875, 0.388726, 0.347536], [0.136821, 0.143952, 0.145229])])(sample_image)

    # print(list(model_inner_resnet.children()))
    # print(model_inner_resnet.fc)
    # print(model_inner_resnet.fc0)
    # # print(model_inner_resnet.layer4[0].conv2)
    # # print(type(model))
    # # print(len(list(model_inner_resnet.children())))
    # # print(list(model.children()))
    # # print(list(list(model.children())[0].children()))

    # img = np.uint8(np.random.uniform(150, 180, (56, 56, 3)))/255
    # img_tensor = torch.unsqueeze(torch.from_numpy(img), 0)

    # full_out = model_inner_resnet.forward(img_tensor)
    # print(full_out)
    # model(img_tensor)
    # activations = SaveFeatures(model_inner_resnet.layer4[0].conv2)
    # print(activations.features)
    # print(type(activations.features))

    # activations.close()
    


if __name__=='__main__':
    main()