import torch
import detectron2

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


def main(input_img_path: str = '/workspaces/detectron/data/hyena.coco/images/train2022/000000003036.jpg'):
    print_versions()
    cfg = get_cfg()
    predictor = load_model(cfg)
    with Image.open(input_img_path) as input_image:
         input_array = np.array(input_image)
         outputs = predictor(input_array)

         v = Visualizer(input_array[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
         img = Image.fromarray(out.get_image()[:, :, ::-1].astype('uint8'), 'RGB')
         img.save('temp_result.png', format="PNG")

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
         

if __name__ == "__main__":
    main()
