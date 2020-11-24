import os
import cv2
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset




def base_transform(image, size, mean ):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform(object):
    def __init__(self, img_size = 300, img_mean = (0, 0, 0)):
        self.img_size = img_size
        self.img_mean = np.array(img_mean, dtype=np.float32)

    def __call__(self, image, boxes = None, labels = None):
        return base_transform(image, self.img_size, self.img_mean), boxes, labels


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



class CustomAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self,):
        pass

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
                loc = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3]+ bbox[1]]
                
                label_idx = obj['category_id']
                
                final_box = list( np.array(loc) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]

        return res 


class CustomDetection(Dataset):
    def __init__(self,
                 root = "./image_folder",
                 json_path = "./annotation.json",
                 transform = None, 
                 target_transform = None):

        self.root = root
        self.coco = COCO(annotation_file= json_path)
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


    def get_class_number(self):
        return len(self.coco.cats)


    def get_class_map(self):

        save_path = os.path.join( os.getcwd(), "label_map.txt")


        print(f"save class map file to { save_path }")
        with open( save_path, "w") as writefile:
            for key in self.coco.cats.keys():
                id_ = self.coco.cats[key]["id"]
                name = self.coco.cats[key]['name']
                print(f"{id_}, {name}", file = writefile)
        

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

        img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
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
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img.transpose(2, 0, 1)), target, height, width

    def get_image(self, idx):
        """Return image object at certain index
        Args:
            idx (int): index
        Return:
            cv2 img
        """
        img_id = self.ids[idx]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img, _, _ = self.transform(img, None, None)
        return img

    def get_annotation(self, index, width, height):
        '''Returns annotation of image at certain index
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        target = self.coco.loadAnns(ann_ids)
        target = self.target_transform(target, width, height)
        return target


if __name__ == '__main__':
    
    dataset = CustomDetection(root = "/Users/kehwaweng/Documents/imageCrawler/Image-Downloader/",
                              json_path = "/Users/kehwaweng/Documents/imageCrawler/Image-Downloader/few_sample/few_sample_dataset.json",
                              transform = BaseTransform(img_size = 300),
                              target_transform = CustomAnnotationTransform())
    loader = DataLoader(dataset= dataset, batch_size = 4, shuffle= True, collate_fn = detection_collate)

    for img, target, in loader:
        print("image shape: {}".format(img.shape))
        for i,t in enumerate(target):
            print(f"{i} of {len(target)} targets with shape: {t.shape}")
        break


