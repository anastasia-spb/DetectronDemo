import torch
import detectron2
import cv2, random, os

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import load_coco_json

from PIL import Image
import numpy as np


def load_model(cfg):
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)

def print_versions():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)


def load_data(annotation_file: str = '/workspaces/DetectronDemo/data/hyena.coco/annotations/instances_train2022.json'):
    return load_coco_json(annotation_file, image_root='/workspaces/DetectronDemo/data/hyena.coco/images/train2022')

def check_gt():
    DatasetCatalog.register("hyenas_dataset", lambda: load_data())
    MetadataCatalog.get("hyenas_dataset").set(thing_classes=["hyenas"])
    hyenas_dataset = MetadataCatalog.get("hyenas_dataset")

    dataset_dicts = load_data()
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=hyenas_dataset, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Check gt", out.get_image()[:, :, ::-1])
        cv2.waitKey()

    
def transfer_from_coco_to_hyenas():
    from detectron2.engine import DefaultTrainer

    DatasetCatalog.register("hyenas_dataset", lambda: load_data())
    MetadataCatalog.get("hyenas_dataset").set(thing_classes=["hyenas"])

    cfg = get_cfg()
    model_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    cfg.DATASETS.TRAIN = ("hyenas_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "hyenas_checkpoint.pth"))
    


def run(input_img_path: str = '/workspaces/DetectronDemo/data/dog.jpg', only_coco: bool = True):
    print_versions()
    cfg = get_cfg()
    

    if only_coco:
        predictor = load_model(cfg)
    else:
        cfg = get_cfg()
        model_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(model_url))
        cfg.DATASETS.TRAIN = ("hyenas_dataset",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 8
        cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []        # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). 
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

    with Image.open(input_img_path) as input_image:
         input_array = np.array(input_image)
         outputs = predictor(input_array)

         v = Visualizer(input_array[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
         img = Image.fromarray(out.get_image()[:, :, ::-1].astype('uint8'), 'RGB')
         img.save('temp_result300.png', format="PNG")

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
         

if __name__ == "__main__":
    run(only_coco=False)
    # check_gt()
    # transfer_from_coco_to_hyenas()
