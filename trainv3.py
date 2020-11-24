import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import tensorflow as tf
import cv2

from data import CustomDetection, CustomAnnotationTransform, detection_collate, BaseTransform, COCODetection, COCOAnnotationTransform
from loss import MultiBoxLossV3
from models import MobileNetv3, SSDMobilenetv3 

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def save_model(save_path, model, epoch):
    if not os.path.exists(save_path):
        os.makedirs( save_path)
    model_name = "model_weights_checkpoint_epoch_" + str(epoch) + ".pt"
    torch.save( model.state_dict(), os.path.join(save_path, model_name))




def main(args):
    
    create_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    save_folder_path = os.path.join(args.save_folder, create_time)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset = CustomDetection(root = args.image_root,
    #                           json_path = args.annotation,
    #                           transform = BaseTransform(img_size = args.image_size),
    #                           target_transform = CustomAnnotationTransform())


    dataset = COCODetection(root = args.image_root,
                            annotation_json = args.annotation,
                            transform = BaseTransform(img_size = args.image_size),
                            target_transform = COCOAnnotationTransform)

    dataloader = DataLoader(dataset= dataset, batch_size = 4, shuffle= True, collate_fn = detection_collate)

    n_classes = dataset.get_class_number() + 1
    print("Detect class number: {}".format(n_classes))

    ## write category id to label name map
    dataset.get_class_map()

    model = MobileNetv3(n_classes = n_classes)
    ssd = SSDMobilenetv3(model, n_classes)

    if args.pretrain_model_path:
        ssd.load_state_dict( torch.load(args.pretrain_model_path))


    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param) 

    optimizer = torch.optim.SGD(params = [{'params': biases, 'lr':  args.learning_rate}, {'params': not_biases}],
                                lr = args.learning_rate, 
                                momentum = args.momentum, 
                                weight_decay = args.weight_decay)   

    ssd = ssd.to(device)
    criterion = MultiBoxLossV3(ssd.priors_cxcy, args.threshold, args.neg_pos_ratio).to(device)

    print(f"epochs: {args.epochs}")
    for param_group in optimizer.param_groups:
        optimizer.param_groups[1]['lr'] = args.learning_rate
    print(f"learning rate. The new LR is {optimizer.param_groups[1]['lr']}")

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode = 'min', 
                                  factor = 0.1, 
                                  patience = 15,
                                  verbose = True, 
                                  threshold = 0.00001, 
                                  threshold_mode = 'rel', 
                                  cooldown = 0, 
                                  min_lr = 0,
                                  eps = 1e-08)

    n_train = min(dataset.__len__(), 5000)
    global_step = 0
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        mean_loss = 0
        inference_count = 0
        ssd.train()
        mean_count = 0
        with tqdm(total = n_train, desc = f"{epoch + 1} / {args.epochs}", unit = 'img') as pbar:
            for img, target in dataloader:
                img = img.to(device)
                # target = [anno.to(device) for anno in target]
                # print(target)
                # boxes = target[:, :-1]
                # labels = target[:, -1]

                boxes = [anno.to(device)[:, :-1] for anno in target]
                labels = [anno.to(device)[:, -1] for anno in target]

                prediction_location_loss, prediction_confidence_loss = ssd(img)
                loss = criterion(prediction_location_loss, prediction_confidence_loss, boxes, labels)
                pbar.set_postfix( **{"loss ": float(loss)})
                mean_loss += float(loss) 
                mean_count += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update( img.shape[0])
                

        scheduler.step(mean_loss)
        writer.add_scalar('Train/Loss', float(mean_loss / mean_count), global_step)
        global_step += 1

        if epoch % 10 == 0 or epoch == args.epochs - 1:   
            save_model(save_folder_path, ssd, epoch)

    writer.close()



def parse_args():
    parser = argparse.ArgumentParser(description='single shot multibox detector with mobilenet v3')

    '''
            dataset
    '''
    parser.add_argument('--image-size',
                        '-s',
                        type = int,
                        default = 300,
                        help = 'image input size ')

    parser.add_argument('--image-root',
                        '-ir', 
                        type = str,
                        default = './image_folder',
                        help = 'use Custom dataset')

    parser.add_argument('--annotation',
                        '-a',
                        type = str,
                        default = "./annotation.json",
                        help = "annotation json file")


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
                        default = 1e-3,
                        help = 'training learning rate')

    parser.add_argument('--momentum',
                        type = float,
                        default = 0.9,
                        help = "gradient descent momentum")

    parser.add_argument('--weight-decay',
                        '-wd',
                        type = float,
                        default = 5e-4,
                        help = 'weight decay for learning rate')

    parser.add_argument('--print-log',
                        default = True,
                        help = "print the loss at the end of each epoch")

    parser.add_argument("--save-folder",
                        type = str,
                        default = "./experience_few_dataset",
                        help = "model weight save path")

    '''
            loss
    '''
    parser.add_argument("--threshold",
                        type = float,
                        default = 0.5,
                        help = 'iou overlap threshold')

    parser.add_argument("--neg-pos-ratio",
                        type = int,
                        default = 3,
                        help = "negative candidate / positive candidate for bbo x class loss")

    '''
            load pretrain model
    '''

    parser.add_argument("--pretrain-model-path",
                        type= str,
                        help = "load pretrain mode;")

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = parse_args()
    main(args)
    print("done")
    


