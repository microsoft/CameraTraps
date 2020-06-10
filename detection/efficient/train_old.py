# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import os
import sys
import json
import argparse
import datetime
import traceback
import yaml
import numpy as np
from tqdm.autonotebook import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

sys.path.append('Yet-Another-EfficientDet-Pytorch/')

from utils_.efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from utils_.efficientdet.loss import FocalLoss
from utils_.calc_eval import calc_mAP_fin, evaluate_mAP

from backbone import EfficientDetBackbone

from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


class Params:
    """
    Docstring:
    Load the project settings from projects/file.yml
    into self.params
    """
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch:' \
                'SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', \
                help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, \
                help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12,  \
                help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, \
                help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,\
                help='whether finetunes only the regressor and the classifier,'\
            'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', \
        help='select optimizer for training, '\
            'suggest using \'admaw\' until the'\
            ' very final stage then switch to \'sgd\'')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, \
        help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, \
        help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0, \
                help='Early stopping\'s parameter: minimum change loss to qualify as improvement')
    parser.add_argument('--es_patience', type=int, default=0,\
                help='Early stopping\'s parameter: number of epochs'\
                     'with no improvement after which training will be stopped.'\
                     'Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', \
        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None, \
        help='whether to load weights from a checkpoint,'\
             'set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=bool, default=True, \
        help='whether visualize the predicted boxes of training,'\
             'the output images will be in test/')
    parser.add_argument('--eval_percent_epoch', type=float, default=10,\
                help=' To determine, at what part of an epoch you want to evaluate.'\
                     'An entry of 10 would mean, for every 1/10th of a training epoch, we evaluate.')
    parser.add_argument('--max_preds_toeval', type=int, default=20000,\
                help=' In the initial phases of training, model produces a lot of'\
                     'bounding boxes for a single image. So, limit the preds '\
                    ' to a certain number to avoid overburning CPU.')
    parser.add_argument('--eval_sampling_percent', type=float, default=10,\
                help=' How much percentage of validation images do you'\
                     ' intend to validate in each epoch.')
    parser.add_argument('--num_visualize_images', type=int, default=10,\
                help=' How much percentage of validation images do you'\
                     ' intend to validate in each epoch.')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    """
    Input : EfficientDet model
    Along with forward function,
    model is mapped with the criterion() here.
    """
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug
        self.evalresults = []
    def forward(self, imgs, annotations, obj_list=None, **kwargs):
        _, regression, classification, anchors = self.model(imgs)
        imgs_scales = kwargs.get('resizing_imgs_scales', None)
        new_ws = kwargs.get('new_ws', None)
        new_hs = kwargs.get('new_hs', None)
        imgs_ids = kwargs.get('imgs_ids', None)
         # `resizing_imgs_scales` will be activated in eval mode
        if new_ws is not None:
            img_max = max(new_ws[0], new_hs[0])
            framed_metas = [(w, h, w/scale, h/scale, img_max-w, img_max-h) \
                           for w, h, scale in zip(new_ws, new_hs, imgs_scales)]
            framed_metas = [(int(a), int(b), int(c), int(d), int(e), int(f)) \
                           for (a, b, c, d, e, f) in framed_metas]
            #Framed metas =  [w,h,org_w,org_h,pad_w,pad_h]
            # Example     = [(512, 384, 2048, 1536, 0, 128)]
            self.evalresults += evaluate_mAP(imgs.detach(), imgs_ids, framed_metas,
                                             regression, classification, anchors)

        imgs_labelled = []
        if self.debug:
            cls_loss, reg_loss, imgs_labelled = self.criterion(classification, regression,
                                                               anchors, annotations,
                                                               imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss, imgs_labelled

def train(opt):
    """
    Input: get_args()
    Function: Train the model.
    """
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    # evaluation json file
    pred_folder = f'{OPT.data_path}/{OPT.project}/predictions'
    os.makedirs(pred_folder, exist_ok=True)
    evaluation_pred_file = f'{pred_folder}/instances_bbox_results.json'

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': True,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                               set=params.train_set,
                               transform=torchvision.transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                         Augmenter(),
                                         Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                          set=params.val_set,
                          transform=torchvision.transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                          Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list),
                                 compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios),
                                 scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except Exception as exception:
            last_step = 0

        try:
            _ = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as rerror:
            print(f'[Warning] Ignoring {rerror}')
            print('[Warning] Don\'t panic if you see this, '\
                  'this might be because you load a pretrained weights with different number of classes.'\
                  ' The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(mdl):
            classname = mdl.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in mdl.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    num_val_iter_per_epoch = len(val_generator)
    # Limit the no.of preds to #images in val. 
    # Here, I averaged the #obj to 5 for computational efficacy
    if opt.max_preds_toeval > 0: 
        opt.max_preds_toeval = len(val_generator)*opt.batch_size* 5 

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iternum, data in enumerate(progress_bar):
                if iternum < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']
                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    if iternum%int(num_iter_per_epoch*(opt.eval_percent_epoch/100)) != 0:
                        model.debug = False
                        cls_loss, reg_loss, _ = model(imgs, annot, obj_list=params.obj_list)
                    else:
                        model.debug = True
                        cls_loss, reg_loss, imgs_labelled = model(imgs, annot, obj_list=params.obj_list)

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iternum + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    if iternum%int(num_iter_per_epoch*(opt.eval_percent_epoch/100)) == 0 and step > 0:
                        # create grid of images
                        imgs_labelled = np.asarray(imgs_labelled)
                        imgs_labelled = torch.from_numpy(imgs_labelled)   # (N, H, W, C)
                        imgs_labelled.transpose_(1, 3) # (N, C, H, W)
                        imgs_labelled.transpose_(2, 3)
                        img_grid = torchvision.utils.make_grid(imgs_labelled)
                        # write to tensorboard
                        writer.add_image('Training_images', img_grid, global_step=step)
#########################################################start EVAL#####################################################
                        model.eval()
                        model.debug = False # Don't print images in tensorboard now.
                        
                        # remove json
                        if os.path.exists(evaluation_pred_file):
                            os.remove(evaluation_pred_file)

                        loss_regression_ls = []
                        loss_classification_ls = []
                        model.evalresults = [] # Empty the results for next evaluation.
                        imgs_to_viz = []
                        num_validation_steps = int(num_val_iter_per_epoch*(opt.eval_sampling_percent/100))
                        for valiternum, valdata in enumerate(val_generator):
                            with torch.no_grad():
                                imgs = valdata['img']
                                annot = valdata['annot']
                                resizing_imgs_scales = valdata['scale']
                                new_ws = valdata['new_w'] 
                                new_hs = valdata['new_h'] 
                                imgs_ids = valdata['img_id']

                                if params.num_gpus >= 1:
                                    imgs = imgs.cuda()
                                    annot = annot.cuda()

                                if valiternum%(num_validation_steps//(opt.num_visualize_images//opt.batch_size)) != 0:
                                    model.debug = False
                                    cls_loss, reg_loss, _ = model(imgs, annot, obj_list=params.obj_list,
                                                                resizing_imgs_scales=resizing_imgs_scales,
                                                                new_ws=new_ws, new_hs=new_hs, imgs_ids=imgs_ids)
                                else:
                                    model.debug = True
                                    cls_loss, reg_loss, val_imgs_labelled = model(imgs, annot, obj_list=params.obj_list,
                                                                            resizing_imgs_scales=resizing_imgs_scales,
                                                                            new_ws=new_ws, new_hs=new_hs, imgs_ids=imgs_ids)
                                    
                                    imgs_to_viz += list(val_imgs_labelled)

                                loss_classification_ls.append(cls_loss.item())
                                loss_regression_ls.append(reg_loss.item())

                            if valiternum > (num_validation_steps):
                                break

                        cls_loss = np.mean(loss_classification_ls)
                        reg_loss = np.mean(loss_regression_ls)
                        loss = cls_loss + reg_loss

                        print(
                            'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                                epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                        writer.add_scalars('Loss', {'val': loss}, step)
                        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                        # create grid of images
                        val_imgs_labelled = np.asarray(imgs_to_viz)
                        val_imgs_labelled = torch.from_numpy(val_imgs_labelled)   # (N, H, W, C)
                        val_imgs_labelled.transpose_(1, 3) # (N, C, H, W)
                        val_imgs_labelled.transpose_(2, 3)
                        val_img_grid = torchvision.utils.make_grid(val_imgs_labelled,nrow=2)
                        # write to tensorboard
                        writer.add_image('Eval_Images', val_img_grid, \
                                         global_step=(step))

                        if opt.max_preds_toeval > 0:
                            json.dump(model.evalresults, open(evaluation_pred_file, 'w'), indent=4)
                            try:
                                val_results = calc_mAP_fin(evaluation_pred_file, \
                                                        val_gt=f'{OPT.data_path}/{OPT.project}/annotations/instances_{params.val_set}.json')

                                for catgname in val_results:
                                    metricname = 'Average Precision  (AP) @[ IoU = 0.50      | area =    all | maxDets = 100 ]'
                                    evalscore = val_results[catgname][metricname]
                                    writer.add_scalars(f'mAP@IoU=0.5 and area=all', {f'{catgname}': evalscore}, step)
                            except Exception as exption:
                                print("Unable to perform evaluation", exption)

                        if loss + opt.es_min_delta < best_loss:
                            best_loss = loss
                            best_epoch = epoch

                            save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                        model.train()

                        # Early stopping
                        if epoch - best_epoch > opt.es_patience > 0:
                            print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                            break
#########################################################EVAL#####################################################


                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as exception:
                    print('[Error]', traceback.format_exc())
                    print(exception)
                    continue
            scheduler.step(np.mean(epoch_loss))
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    """
    Save the model
    """
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(OPT.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(OPT.saved_path, name))


if __name__ == '__main__':
    OPT = get_args()
    train(OPT)
