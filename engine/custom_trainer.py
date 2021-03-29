from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
import detectron2.data.build as build

from ..data.custom_dataset_mapper import RotatedDatasetMapper


class RotatedTrainer(DefaultTrainer):
    """
      The trainer for rotated box detection task    
    """
    @classmethod
    def build_train_loader(cls, cfg):
        mapper=RotatedDatasetMapper(cfg)
        res=build._train_loader_from_config(cfg, mapper=mapper)
        dataset = res["dataset"]
        return build_detection_train_loader(dataset=dataset, mapper=mapper, total_batch_size=cfg.SOLVER.IMS_PER_BATCH)


    
