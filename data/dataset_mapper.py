from detectron2.data import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.structures import BoxMode

import torch
import numpy as np
import copy

class RotatedDatasetMapper(DatasetMapper):
    """
        A dataset mapper do the same thing as DatasetMapper for detection task, except for 
        loading the bbox with mode BoxMode.XYWHA_ABS instead of BoxMode.XYXY_ABS
    """

    def transform_instance_annotations_rotated(
        self, annotation, transforms, image_size
    ):
        if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)
        
        bbox = annotation["bbox"]
        alpha = bbox[-1]
        bbox = bbox[:4]
        bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
        bbox = bbox.tolist()
        bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        bbox = np.minimum(bbox, list(image_size + image_size)[::-1])
        bbox = np.append(bbox, alpha)
        annotation["bbox"] = bbox
        annotation["bbox_mode"] = BoxMode.XYWHA_ABS

        return annotation

    def __call__(self, dataset_dict):

        dataset_dict=copy.deepcopy(dataset_dict)
        image=utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input=T.AugInput(image, sem_seg=None)
        transforms=self.augmentations(aug_input)

        image=aug_input.image

        image_shape=image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:

            annos = [
                self.transform_instance_annotations_rotated(
                    obj, transforms, image_shape
                )
                # obj 
                for obj in dataset_dict.pop("annotations") 
                if obj.get("iscrowd", 0) == 0
            ]

            instances=utils.annotations_to_instances_rotated(
                annos, image_shape
            )

            if self.recompute_boxes:
                instances.gt_boxes=instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
