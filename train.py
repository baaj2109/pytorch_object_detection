import os
import numpy as np
import argparse
import time
from tqdm import tqdm

from data import COCODetection, SSDAugmentation, COCOAnnotationTransform, detection_collate, BaseTransform
from loss import MultiBoxLoss
from models import mobilenetv2, create_mobilenetv2_ssd_lite, SSD, PriorBox
from config import MOBILEV2_512


import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0

def save_model(save_path, model, epoch):
    if not os.path.exists(save_path):
        os.makedirs( save_path)
    model_name = "model_weights_checkpoint_epoch_" + str(epoch) + ".pt"
    torch.save( model.state_dict(), os.path.join(save_path, model_name))

def train(args):
    create_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    save_folder_path = os.path.join(args.save_folder, create_time)

    n_classes = [20, 80][args.dataset == 'COCO']

    model = mobilenetv2(n_classes = n_classes,
                         width_mult = args.width_multi,
                         round_nearest = 8, 
                         dropout_ratio = args.dropout_ratio,
                         use_batch_norm = True,)

    ssd = create_mobilenetv2_ssd_lite(model, 
                                      n_classes,
                                      width_mult = args.width_multi,  
                                      use_batch_norm = True)
    print("builded ssd module")

    optimizer = optim.Adam(ssd.parameters(),
                           lr = args.learning_rate, 
                           weight_decay = args.weight_decay)
    criterion = MultiBoxLoss(n_classes,
                             overlap_thresh = args.overlap_threshold,
                             prior_for_matching = True,
                             bkg_label = 0,
                             neg_mining = True,
                             neg_pos = args.neg_pos_ratio,
                             neg_overlap = 0.5,
                             encode_target = False)
    with torch.no_grad():
        prior_box = PriorBox(MOBILEV2_512)
        priors = Variable(prior_box.forward())
        print("created default bbox")

    train_dataset = COCODetection(root = args.root,
                                  image_set = args.train_image_folder, 
                                  transform = SSDAugmentation(img_size = args.image_size),
                                  target_transform = COCOAnnotationTransform())

    train_dataloader = DataLoader(dataset = train_dataset, 
                                  batch_size = args.batch_size,
                                  shuffle = True, 
                                  collate_fn = detection_collate)

    # val_dataset = COCODetection(root = args.root,
    #                             image_set = args.val_image_folder,
    #                             transfrom = BaseTransform(img_size = args.img_size),
    #                             target_transform = COCOAnnotationTransform())


    n_train = min(train_dataset.__len__(), 5000)
    # n_val = val_dataset.__len__()
    global_step = 0
    writer = SummaryWriter()
    for epoch in range(args.epochs):
        ssd.train()
        mean_loss_conf = 0
        mean_loss_loc = 0

        with tqdm(total = n_train, desc = f"{epoch + 1} / {args.epochs}", unit = 'img') as pbar:
            for img, target in train_dataloader:
                
                img = Variable(img)
                target = [Variable(anno) for anno in target]
                optimizer.zero_grad()

                inference = ssd(img)

                loss_loc, loss_conf = criterion( inference, priors, target)
                writer.add_scalar('Train/location_loss', float(loss_loc), global_step)
                writer.add_scalar('Train/confidence_loss', float(loss_conf), global_step)

                pbar.set_postfix( **{"location loss ": float(loss_loc),
                                     "confidence loss ": float(loss_conf)})
                
                mean_loss_loc += float(loss_loc)
                mean_loss_conf += float(loss_conf)

                total_loss = loss_loc + loss_conf
                total_loss.backward()
                
                # # clip gradient
                # # clip_grad_norm_(net.parameters(), 0.1)

                # optimizer.step()
                pbar.update( img.shape[0])
                global_step += 1
                

        save_model(save_folder_path, ssd, epoch)

        # with tqdm(total = n_val, desc = "Validation", unit = "img", leave = False) as vpbar:
        #     for i in range(n_val):
        #         img = val_dataset.pull_img(i)


    writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description='single shot multibox detector with mobilenet v2')

    '''
            loss
    '''
    parser.add_argument("--overlap_threshold",
                        type = float,
                        default = 0.5,
                        help = 'iou overlap threshold')

    parser.add_argument("--neg-pos-ratio",
                        type = int,
                        default = 3,
                        help = "negative candidate / positive candidate for bbo x class loss")


    '''
            model
    '''
    parser.add_argument('--width-multi',
                        type = float,
                        default = 1.0,
                        help = "multiply value to extent model width")

    parser.add_argument('--dropout_ratio',
                        type = float,
                        default = 0.2,
                        help = "percentage of drop out ratio")

    '''
            dataset
    '''
    parser.add_argument('--image-size',
                        '-s',
                        type = int,
                        default = 512,
                        help = 'image input size , TODO: support image size 300')

    parser.add_argument('--dataset',
                        '-d', 
                        type = str,
                        default = 'COCO',
                        help = 'use COCO dataset, TODO: support voc dataset')

    parser.add_argument('--root',
                        type = str,
                        default = "./",
                        help = "path to coco dataset folder")

    parser.add_argument('--train-image-folder',
                        type = str, 
                        default = "train2017",
                        help = 'path to train image')

    parser.add_argument('--val-image-folder',
                        type = str,
                        default = 'val2017',
                        help = 'path to validation image')

    parser.add_argument('--coco-label',
                        type = str,
                        default = './coco_labels.txt',
                        help = "label file contain two index and class name" \
                               "file format origional coco index, relabel index, class name >> 1,1,person ")

    '''
            training 
    '''
    parser.add_argument('--batch_size',
                        '-b',
                        type = int,
                        default = 16,
                        help = "training batch size")

    parser.add_argument('--epochs',
                        '-ep',
                        type = int,
                        default = 100,
                        help = 'training epoch')

    parser.add_argument('--learning-rate',
                        '-lr',
                        type = float,
                        default = 4e-3,
                        help = 'training learning rate')

    parser.add_argument('--weight-decay',
                        '-wd',
                        type = float,
                        default = 1e-4,
                        help = 'weight decay for learning rate')

    parser.add_argument('--print-log',
                        default = True,
                        help = "print the loss at the end of each epoch")

    parser.add_argument("--save-folder",
                        type = str,
                        default = "./experience",
                        help = "model weight save path")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    train(args)
    print("done")



