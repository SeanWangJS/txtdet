import glob
import os

from PIL import Image

from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

__all__ = ["load_msra_td500", "register_msra_td500"]

def load_msra_td500(dirname: str, split: str):
    """
        Load MSRA_TD500 text detection data

        Args:
            dirname: 
            split: "train" or "test"
    """

    paths=glob.glob(dirname + "/" + split + "/*.JPG")
    dicts=[]
    
    for path in paths:
        gt_path=os.path.splitext(path)[0] + ".gt"
        with open(gt_path) as f:
            lines = f.readlines()
            annotations = []
            for line in lines:
                ss=line.split(" ")
                angle = float(ss[6])
                angle = - angle * 90 ## to ccw degree
                left=float(ss[2])
                top = float(ss[3])
                w = float(ss[4])
                h = float(ss[5])
                x = left + w / 2
                y = top + h / 2
                bbox = [x, y, w, h, angle]
                annotation = {
                    "category_id": 0,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWHA_ABS
                }
                annotations.append(annotation)
        
        img=Image.open(path)
        dicts.append({
            "file_name": path,
            "width": img.width,
            "height": img.height,
            "annotations": annotations
        })
    
    return dicts

def register_msra_td500(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_msra_td500(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes="text", dirname=dirname, split=split
    )