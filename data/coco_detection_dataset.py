import os
import cv2
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from .data_augmentation import SSDAugmentation


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets



class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, path = './coco_labels.txt'):
        self.label_map = self._get_label_map(path)

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [xmin, ymin, xmax, ymax, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                pass
                # print("no bbox error!")

        return res 
    
    def _get_label_map(self, label_file):
        label_map = {}
        labels = open(label_file, 'r')
        for line in labels:
            ids = line.split(',')
            label_map[int(ids[0])] = int(ids[1])
        return label_map



class COCODetection(Dataset):
    def __init__(self,
                 root = "./coco_dataset/",
                 image_set = "train2017",
                 transform = SSDAugmentation(), 
                 target_transform = COCOAnnotationTransform()):
        """COCO datset for object detection 
        Args:
            root (str): path to coco dataset folder, all coco folder under this path
            image_set (str): image set choice from [train2017 , val2017]
            transform (object): image augment function zoo
            target_transform (object) : function for process target bbox and label
        """
        self.root = root
        self.image_folder = os.path.join( root, image_set)
        self.coco = COCO(annotation_file= os.path.join(root,
                                                       "annotations_2017",
                                                       "instances_{}.json".format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by coco.loadAnns
        """
        img, gt, h, w = self.pull_item(idx)
        return img, gt
    
    def __len__(self): 
        return len(self.ids)
    
    def pull_item(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: tuple (image, target, width, height)
                    target is the object returned by coco.loadAnns
        """
        img_id = self.ids[idx]
        target = self.coco.imgToAnns[img_id]
#         ann_ids = self.coco.getAnnIds(img_id)
#         target = self.coco.loadAnns(ann_ids)
        img_path = os.path.join(self.image_folder, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(img_path), "loading image error"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, 
                                                target[:, :4],
                                                target[:, 4])
        
        return torch.from_numpy(img.transpose(2, 0, 1)), target, height, width

    def get_image(self, idx):
        """Return image object at certain index
        Args:
            idx (int): index
        Return:
            cv2 img
        """
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(os.path.join(self.image_folder, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns annotation of image at certain index
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        target = self.target_transform(target, width, height)
        return self.coco.loadAnns(ann_ids)




if __name__ == '__main__':
    dataset = COCODetection(root = "./coco_dataset/",
                            image_set = "train2017",
                            transform = SSDAugmentation(),
                            target_transform = COCOAnnotationTransform())
    loader = DataLoader(dataset= dataset, batch_size = 4, shuffle= True, collate_fn = detection_collate)

    for img, target, in loader:
        print("image shape: {}".format(img.shape))
        for i,t in enumerate(target):
            print(f"{i} of {len(target)} targets with shape: {t.shape}")
        break





