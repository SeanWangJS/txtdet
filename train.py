from detectron2.data import DatasetCatalog 
from detectron2.data import MetadataCatalog 

from detectron2.data.datasets.pascal_voc import load_voc_instances

from detectron2.config import get_cfg

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from data.msra_td500 import register_msra_td500
from engine.custom_trainer import RotatedTrainer

register_msra_td500("msra-rd500-train", "/opt/dataset/MSRA-TD500", "train")
register_msra_td500("msra-rd500-test", "/opt/dataset/MSRA-TD500", "test")

cfg= get_cfg()
cfg.OUTPUT_DIR ="./output_text"
config_file = model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
weight_file = model_zoo.get_checkpoint_url("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
cfg.merge_from_file(config_file)


cfg.DATASETS.TRAIN = ("msra-rd500-train", )
cfg.DATASETS.TEST = ("msra-rd500-test", )

cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128
cfg.MODEL.WEIGHTS = weight_file
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.PROPOSAL_GENERATOR.NAME="RRPN"
cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)
cfg.MODEL.ANCHOR_GENERATOR.NAME="RotatedAnchorGenerator"
cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[[-90, -60, -30, 0, 30, 60, 90]]

cfg.MODEL.ROI_HEADS.NAME="RROIHeads"
cfg.MODEL.ROI_BOX_HEAD.NAME="FastRCNNConvFCHead"
cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (1.0,1.0,1.0,1.0,1.0)
cfg.MODEL.ROI_BOX_HEAD.NUM_FC=1

print(cfg)

trainer = RotatedTrainer(cfg)
trainer.resume_or_load(False)
trainer.train()