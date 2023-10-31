from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Model configuration
model_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
output_dir = '/workspaces/detectron/output'

# Dataset configuration
tag = "hyenas_dataset"
thing_classes=["hyena"]
image_root = '/workspaces/detectron/data/hyena.coco/images/train2022'
annotation_file = '/workspaces/detectron/data/hyena.coco/annotations/instances_train2022.json'

def setup_model(model_weights: str = None, threshold: float = 0.7):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.OUTPUT_DIR = output_dir
    if model_weights is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url) 
    else:
        MetadataCatalog.get(tag).set(thing_classes=thing_classes)
        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.DATASETS.TRAIN = (tag,)
    return cfg


def update_config_for_train(cfg):
    cfg.DATASETS.TRAIN = (tag,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    return cfg