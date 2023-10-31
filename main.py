import torch
import detectron2
import cv2, random, os

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import load_coco_json

from PIL import Image
import numpy as np

import config as config


def print_versions():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)


def load_data():
    return load_coco_json(config.annotation_file, config.image_root)

def check_gt():
    DatasetCatalog.register(config.tag, lambda: load_data())
    MetadataCatalog.get(config.tag).set(thing_classes=config.thing_classes)
    hyenas_dataset = MetadataCatalog.get(config.tag)

    dataset_dicts = load_data()
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=hyenas_dataset, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Check gt", out.get_image()[:, :, ::-1])
        cv2.waitKey()

    
def transfer_from_coco_to_hyenas():
    from detectron2.engine import DefaultTrainer

    DatasetCatalog.register(config.tag, lambda: load_data())
    MetadataCatalog.get(config.tag).set(thing_classes=config.thing_classes)

    cfg = config.setup_model()
    cfg = config.update_config_for_train(cfg)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


def run(input_img_path: str, only_coco: bool = True):
    print_versions()
    
    if only_coco:
        cfg = config.setup_model()
        predictor = DefaultPredictor(cfg)
    else:
        cfg = config.setup_model(model_weights=os.path.join(config.output_dir, 'model_final.pth'))
        predictor = DefaultPredictor(cfg)

    with Image.open(input_img_path) as input_image:
         input_array = np.array(input_image)
         outputs = predictor(input_array)

         v = Visualizer(input_array[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
         img = Image.fromarray(out.get_image()[:, :, ::-1].astype('uint8'), 'RGB')
         filename, _ = os.path.splitext(input_img_path)
         filename += '_coco_result.png' if only_coco else '_result.png'
         img.save(filename, format="PNG")

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
         

if __name__ == "__main__":
    run(input_img_path = '/workspaces/detectron/data/test.jpg', only_coco=False)
    # check_gt()
    # transfer_from_coco_to_hyenas()
