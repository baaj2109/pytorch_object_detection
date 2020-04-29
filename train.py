import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import tensorflow as tf


from data import COCODetection, SSDAugmentation, COCOAnnotationTransform, detection_collate, BaseTransform
from loss import MultiBoxLoss
from models import mobilenetv2, create_mobilenetv2_ssd_lite, SSD, PriorBox
from config import MOBILEV2_512, MOBILEV2_300

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


GPU = False
if torch.cuda.is_available():
    GPU = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')



def save_model(save_path, model, epoch):
    if not os.path.exists(save_path):
        os.makedirs( save_path)
    model_name = "model_weights_checkpoint_epoch_" + str(epoch) + ".pt"
    torch.save( model.state_dict(), os.path.join(save_path, model_name))


def load_tf_weights(args, state_dict):
    tf_vars = []
    ckpt_path = args.pretrain_tfmodel
    init_vars = tf.train.list_variables(ckpt_path)
    tf_weight = {}
    for name, shape in init_vars:
        array = tf.train.load_variable(ckpt_path, name)
        tf_vars.append((name, array.squeeze()))
        tf_weight[name] = array.squeeze()

    tensorflow_weights_list = []
    with open(args.pretrain_tfmodel_weight_list, "r") as readfile:
        for line in readfile.readlines():
            tensorflow_weights_list.append(line.strip().split(",")[0].replace('\'',"").replace("(",""))
    while( "" in tensorflow_weights_list):
        tensorflow_weights_list.remove("")

    tf_index = 0
    for i in state_dict.keys():
        if "num_batches_tracked" in i: continue            
        np_weight = tf_weight[tensorflow_weights_list[tf_index]]
        
        target_shape = state_dict[i].shape
        if "/weights" in tensorflow_weights_list[tf_index]:
            if len(np_weight.shape) != 4:
                np_weight = np.expand_dims(np_weight, 0)
                np_weight = np.expand_dims(np_weight, 0)
            np_weight = np_weight.transpose(3, 2, 0, 1)
        
        elif "/depthwise_weights" in tensorflow_weights_list[tf_index]:
            np_weight = np.expand_dims(np_weight, -1)
            np_weight = np_weight.transpose(2, 3, 0, 1)

        else:
            np_weight = np_weight
            
        assert target_shape == np_weight.shape
        state_dict[i] = torch.from_numpy(np_weight)
        
        tf_index += 1

    return state_dict




def train(args):
    create_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    save_folder_path = os.path.join(args.save_folder, create_time)

    # n_classes = [20, 80][args.dataset == 'COCO']
    n_classes = 91

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

    if GPU:
        import torch.backends.cudnn as cudnn
        model.cuda()
        ssd.cuda()
        cudnn.benchmark = True

    if args.pretrain_model:
        ssd.load_state_dict(torch.load(args.pretrain_model, 
                            map_location = torch.device('cpu')))

    elif args.pretrain_tfmodel and args.pretrain_tfmodel_weight_list:
        ssd_state_dict = ssd.state_dict()
        tf_weights_dict = load_tf_weights(args, ssd_state_dict)
        ssd.load_state_dict( tf_weights_dict)


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
        # prior_box = PriorBox(MOBILEV2_512)
        prior_box = PriorBox(MOBILEV2_300)
        priors = Variable(prior_box.forward())
        print("created default bbox")

    train_dataset = COCODetection(root = args.root,
                                  image_set = args.train_image_folder, 
                                  transform = SSDAugmentation(img_size = args.image_size),
                                  target_transform = COCOAnnotationTransform(args.coco_label))

    train_dataloader = DataLoader(dataset = train_dataset, 
                                  batch_size = args.batch_size,
                                  shuffle = True, 
                                  collate_fn = detection_collate)

    val_dataset = COCODetection(root = args.root,
                                image_set = args.val_image_folder,
                                transform = BaseTransform(img_size = args.image_size),
                                target_transform = COCOAnnotationTransform())


    n_train = min(train_dataset.__len__(), 5000)
    n_val = min(val_dataset.__len__(), 1000)
    global_step = 0
    val_global_step = 0
    writer = SummaryWriter()
    for epoch in range(args.epochs):
        mean_loss_conf = 0
        mean_loss_loc = 0
        inference_count = 0

        ssd.train()
        with tqdm(total = n_train, desc = f"{epoch + 1} / {args.epochs}", unit = 'img') as pbar:
            for img, target in train_dataloader:
                
                if GPU:
                    img = Variable(img.cuda())
                    target = [Variable(anno.cuda()) for anno in target]
                else:
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

                optimizer.step()
                pbar.update( img.shape[0])
                global_step += 1
                inference_count += img.shape[0]

                if inference_count > n_train: break
            pbar.set_postfix( **{"location loss ": float(mean_loss_loc / n_train),
                                 "confidence loss ": float(mean_loss_conf / n_train)})

        ssd.eval()
        val_mean_loss_loc = 0
        val_mean_loss_conf = 0
        with tqdm(total = n_val, desc = "Validation", unit = "img", leave = False) as vpbar:
            for i in range(n_val):
                img = val_dataset.get_image(i)
                height, width, _ = img.shape()
                target = val_dataset.get_annotation(i, width, height)

                if GPU:
                    img = torch.from_numpy( np.expand_dims( 
                        img.transpose(2, 0, 1), 0)).to(dtype = torch.float32).cuda()
                    target = torch.FloatTensor(target).unsqueeze(0).cuda()
                else:
                    img = torch.from_numpy( np.expand_dims( 
                        img.transpose(2, 0, 1), 0)).to(dtype = torch.float32)
                    target = torch.FloatTensor(target).unsqueeze(0)

                inference = ssd(img)
                loss_loc, loss_conf = criterion(inference, prior, target)
                
                val_mean_loss_loc += float(loss_loc)
                val_mean_loss_conf += float(loss_conf)
                vpbar.set_postfix( **{'Validation location loss': float(loss_loc),
                                     'confidnece loss': flaot(loss_conf)})
                vpbar.update(1)

            vpbar.set_postfix( **{'Validation location loss': float(val_mean_loss_loc / n_val),
                                 'confidnece loss': float(val_mean_loss_conf / n_val)})
            writer.add_scalar('Test/location_loss', float(val_mean_loss_loc / n_val), val_global_step)
            writer.add_scalar('Test/confidence_loss', float(val_mean_loss_conf/ n_val), val_global_step)

        save_model(save_folder_path, ssd, epoch)
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

    '''
            load pretrain model 
    '''
    parser.add_argument("--pretrain-model",
                        type = str,
                        help = "pretrain model path")

    parser.add_argument('--pretrain-tfmodel',
                        type = str,
                        help = "pretrain tensorflow model path : model.ckpt")

    parser.add_argument('--pretrain-tfmodel-weight-list',
                        type = str,
                        help = "pretrain tensorflow model weight list compare to pytorch model weight list")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    train(args)
    print("done")




'''
commend log 
0417
python train.py --root /Volumes/IPEVO_X0244/coco_dataset/

'''

