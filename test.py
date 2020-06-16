import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from pycocotools.coco import COCO
import argsparse

               
from models import mobilenetv2, create_mobilenetv2_ssd_lite, PriorBox, \
                   mobilenetv3, create_mobilenetv3_ssd_lite, Detect
from config import  MOBILEV2_300, MOBILEV3_300

from box_utils import decode


def check_output_folder_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_prior(args):
    with torch.no_grad():
        if args.model == "mobilenetv2":
            prior_box = PriorBox(MOBILEV2_300)

        elif args.model == "mobilenetv3":
            prior_box = PriorBox(MOBILEV3_300)

        priors = Variable(prior_box.forward())
        print("created default bbox")
    return priors


def detect(args):


    coco = COCO(annotation_file = args.json_file_path)
    n_classes = len(coco.cats)

    ## load model 
    if args.model == "mobilenetv2":
        model = mobilenetv2(n_classes = n_classes,
                             width_mult = args.width_mult,
                             round_nearest = 8, 
                             dropout_ratio = args.dropout_ratio,
                             use_batch_norm = True,)

        ssd = create_mobilenetv2_ssd_lite(model, 
                                          n_classes,
                                          width_mult = args.width_mult,  
                                          use_batch_norm = True)

    elif args.model == "mobilenetv3":
        model = mobilenetv3(model_mode = args.model_mode,
                            n_classes = n_classes,
                            width_mult = args.width_mult,
                            dropout_ratio = args.dropout_ratio)

    ## load data
    ids = list(coco.imgToAnns.keys())
    img_id = random.sample(ids, 1)[0]

    if args.image_path:
        img = cv2.imread(args.image_path)
        img = cv2.cvtColor(img, cv2.BGR2RGB)       

    else:
        image_path = coco.loadImgs(img_id)[0]['file_name']

        img = cv2.imread( os.path.join(args.root, image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = coco.getAnnsIds(imgIds = img_id)
        target = coco.loadAnns(ann_ids)

        ## save target
        if args.show_target :
            fig, ax = plt.subplot(1)
            ax.imshow(img)
            for t in target:
                bbox = t['bbox']
                rect = patches.Rectangle((bbox[0], bbox[1]),
                                         bbox[2],
                                         bbox[3], 
                                         linewidth = 1, 
                                         edgecolor = 'r', 
                                         facecolor = 'none')
                ax.add_patch(rect)

            check_output_folder_exist(args.output_path)
            plt.savefig(args.output_path, "target.png")



    x = img.copy()
    x = cv2.resize(x, (args.image_size, image_size))
    x = torch.from_numpy( np.expand_dims( x.transform(2, 0, 1), 0)).to(dtype = torch.float32)
    ssd.eval()
    o = ssd(x)
    detector = Detect(n_classes, 0, MOBILEV3_300) if args.model == "mobilenetv3" else Detect(n_classes, 0, MOBILEV2_300)
    prior = get_prior(args)
    boxes, scores = detector.forward(o, prior)

    bboxe = boxes.detach().numpy()[0]
    score = scores.detach().numpy()[0]

    list_ = []
    with open( os.path.join(args.output_path, "log.txt"), "w") as writeFile:
        print("bbox format : [xmin, ymin, xmax, ymax]")
        for i,v in enumerate(np.argmax(score, axis = 1)):
            if v != 0:
                # print(i, v, coco.cats[v]['name'] ,score[i][v])
                print(f"box index: {i}, class : {v} {coco.cats[v]['name']}, confidence: {score[i][v]}", file = writeFile)
                list_.append(i)

    if args.show_target:
        height, width, _ = img.shape
        fig,ax = plt.subplots(1)
        ax.imshow(img)

        print(list_)
        for l in list_:
            b = bboxe[l]
            loc = [b[0] * width, b[1] * height, b[2] * width, b[3] * height]
            print(loc)
            rect = patches.Rectangle((loc[0], loc[1]),
                                     loc[2] - loc[0], 
                                     loc[3] - loc[1], 
                                     linewidth = 1, 
                                     edgecolor = 'r', 
                                     facecolor = 'none')
            ax.add_patch(rect)
            
        check_output_folder_exist(args.output_path)
        plt.savefig(args.output_path, "detect.png")



def parse_args():
    parser = argparse.ArgumentParser(description='test object detection ')

    parser.add_argument("--root",
                        type = str,
                        default = "./",
                        help = "image root of annotation image path ")

    parser.add_argument("--json-file-path",
                        type = str,
                        default = "./annotation.json",
                        help = 'path to annotation json file')

    parser.add_argument("--show-target",
                        action = "store_true",
                        default = False,
                        help = "show detection target must have json file")

    parser.add_argument("--output-path",
                        type = str,
                        default = "./test_detection_log",
                        help = "detection result path")

    parser.add_argument("--image-size",
                        type = int,
                        default = 300,
                        help = "model input size")

    parser.add_argument("--image-path",
                        type = str,
                        default = None,
                        help = "image path")

    '''
            model
    '''
    parser.add_argument("--model",
                        type = str,
                        choices = ["mobilenetv2", "mobilenetv3"],
                        default = "mobilenetv2",
                        help = "select basic model structure")

    parser.add_argument("--model-mode",
                        type = str,
                        choices = ["LARGE", "SMALL"],
                        default = "LARGE",
                        help = "only use for mobile net v3 structure")

    parser.add_argument('--width-mult',
                        type = float,
                        default = 1.0,
                        help = "multiply value to extent model width")

    parser.add_argument('--dropout-ratio',
                        type = float,
                        default = 0.2,
                        help = "percentage of drop out ratio")

    parser.add_argument("--pretrain-path",
                        type = str,
                        default = "./pretrain_path.pt",
                        help = "pretrain model path")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    detect(args)




