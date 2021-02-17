# Do some pip install stuff here in colab

import os
import json
import urllib.request

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

import numpy as np
import matplotlib.pyplot as plt

from pycocotools import mask

from segments.utils import export_dataset


class Model:
    def __init__(self, predictor):
        self.predictor = predictor

    def _convert_to_segments_format(self, image, outputs):
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
        annotations = []
        counter = 1
        instances = outputs['instances']
        for i in range(len(instances.pred_classes)):
            category_id = int(instances.pred_classes[i])
            instance_id = counter
            mask = instances.pred_masks[i].cpu()
            segmentation_bitmap[mask] = instance_id
            annotations.append({'id': instance_id, 'category_id': category_id})
            counter += 1
        return segmentation_bitmap, annotations

    def __call__(self, image):
        image = np.array(image)
        outputs = self.predictor(image)
        label, label_data = self._convert_to_segments_format(image, outputs)

        return label, label_data
    

def train_model(dataset):
    # Export the dataset to COCO format
    export_file, image_dir = export_dataset(dataset, export_format='coco-instance')
    
    # Register it as a COCO dataset in the Detectron2 framework
    try:
        register_coco_instances('my_dataset', {}, export_file, image_dir)
    except:
        print('Dataset was already registered')
    dataset_dicts = load_coco_json(export_file, image_dir)
    MetadataCatalog.get('my_dataset').set(thing_classes=[c['name'] for c in dataset.categories])
    segments_metadata = MetadataCatalog.get('my_dataset')
    print(segments_metadata)
    
    # Configure the training run
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('my_dataset',)
    cfg.DATASETS.TEST = ()
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2 # 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset.categories)  # number of categories
#     cfg.MODEL.DEVICE = 'cuda'

    # Start the training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Return the model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = ('my_dataset', )
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    predictor = DefaultPredictor(cfg)
    model = Model(predictor)

    return model
    
    
def get_image_urls(topic):
    with open('{}.json'.format(topic)) as json_file:
        image_urls = json.load(json_file)
    return image_urls


def visualize(*args):
    images = args
    for i, image in enumerate(images):
        plt.subplot(1,len(images),i+1)
        plt.imshow(np.array(image))
    plt.show()
