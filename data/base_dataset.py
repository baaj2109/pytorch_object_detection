import os
import cv2
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class CocoBaseDataset(Dataset):

    def __init__(self, image_root, annotation_json, transform, target_transform):
        self.image_root = image_root
        self.annotation_json = annotation_json
        self.transform = transform
        self.target_transform = target_transform
        self.coco = COCO(annotation_file = annotation_json)
        self.ids = list(self.coco.imgToAnns.keys())

    def __getitem__(self, idx):
        img, gt, h, w = self.pull_item(idx)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def get_class_size(self):
        return len(self.coco.cats)

    def get_class_map(self, save_path):
        save_path = os.path.join(save_path, "label_map.txt")

        print(f"save class map file to {save_path}")
        with open(save_path, "w") as writefile:
            for key in self.coco.cats.keys():
                id_ = self.coco.cats[key]["ids"]
                name = self.coco.cats[key]["name"]
                print(f"{id_}, {name}", file = writefile)

    def pull_item(self, idx):
        """
        Args:
            idx (int): index
        Return: 
            img (torch.tensor): image with shape(1, c, h, w)
            gt (list): list with 5 element [xmin, ymin, width, height]
            h (int): height of image
            w (int): width of image
        """
        img_id = self.ids[idx]
        target = self.coco.imgToAnns[img_id]

        img_path = os.path.join( self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        try:
            img = cv2.imread(img_path)

        except FileNotFoundError:
            print(f"File: {img_path} not exist")
            return 

        finally:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape

            if self.target_transform is not None:
                target = self.target_transform(target, width, height)

            if self.transform is not None:
                target = np.array(target)
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                target = np.hstack(( boxes, np.expand_dims( labels, axis = 1)))
        return torch.from_numpy( img.transpose(2, 0, 1)), target, height, width 


    def get_image(self, idx):
        """
        Args:
            idx (int): index
        Return:
            img (numpy): image
        """
        img_id = self.ids[idx]
        imag_path = self.coco.loadImgs(img_id)[0]['file_name']
        try:
            img = cv2.imread(os.path.join(self.root, img_apth))

        except FileNotFoundError:
            print(f"File: {img_path} not exist")
            return 

        finally:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_annotation(self, idx, width, height):
        """
        Args: 
            idx (int): index
        Return:
            target (list): 2d list contain label, bbox
                         [label, xmin, ymin, xmax, ymax]
        """
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        target = self.coco.loadAnns(ann_ids)
        target = self.target_transform(target, width, height)
        return target


    








