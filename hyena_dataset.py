'''
https://github.com/ppwwyyxx/cocoapi
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/

https://github.com/facebookresearch/detectron2/blob/898507047cf441a1e4be7a729270961c401c4354/detectron2/data/datasets/coco.py#L500
'''

from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog


def load_data(annotation_file: str = '/workspaces/detectron/data/hyena.coco/annotations/instances_train2022.json'):
    hyena_train_dicts = load_coco_json(annotation_file, image_root='/workspaces/detectron/data/hyena.coco/images/train2022')
    print("Done loading {} samples.".format(len(hyena_train_dicts)))
    DatasetCatalog.register("hyenas_dataset", hyena_train_dicts)
    MetadataCatalog.get("hyenas_dataset").set(thing_classes=["hyenas"])
    return MetadataCatalog.get("hyenas_dataset")


if __name__ == "__main__":
    hyenas_dataset = load_data()