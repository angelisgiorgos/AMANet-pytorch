# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import torch
from fvcore.common.file_io import PathManager
import os
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from PIL import Image
import numpy as np
from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
import cv2

class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapperper`
    """

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.densepose_on   = cfg.MODEL.DENSEPOSE_ON
        self.dp_segm_on = cfg.MODEL.ROI_DENSEPOSE_HEAD.SEMSEG_ON
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.densepose_on:
            densepose_transform_srcs = [
                MetadataCatalog.get(ds).densepose_transform_src
                for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            ]
            assert len(densepose_transform_srcs) > 0
            # TODO: check that DensePose transformation data is the same for
            # all the datasets. Otherwise one would have to pass DB ID with
            # each entry to select proper transformation data. For now, since
            # all DensePose annotated data uses the same data semantics, we
            # omit this check.
            print(densepose_transform_srcs[0], cfg.DATASETS.TRAIN)
            print(densepose_transform_srcs)
            densepose_transform_data_fpath = PathManager.get_local_path(densepose_transform_srcs[0])
            self.densepose_transform_data = DensePoseTransformData.load(
                '/data/angelisg/datasets/DensePose_COCO/UV_data/UV_symmetry_transforms.mat'
            )

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # USER: Don't call transpose_densepose if you don't need
        annos = [
            self._transform_densepose(
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ),
                transforms,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape)

        if len(annos) and "densepose" in annos[0]:
            gt_densepose = [obj["densepose"] for obj in annos]
            instances.gt_densepose = DensePoseList(gt_densepose, instances.gt_boxes, image_shape)
        if self.dp_segm_on:
            dp_seg_image_root = './datasets/coco2014/dp_seg_images'
            ori_image_dir = dataset_dict["file_name"].split('/')
            dp_seg_image_dir = os.path.join(dp_seg_image_root,ori_image_dir[-2], ori_image_dir[-1])
            with PathManager.open(dp_seg_image_dir, "rb") as f:
                dp_seg_image = Image.open(f)
                dp_seg_image = np.asarray(dp_seg_image, dtype="uint8")
            # dp_seg_image = utils.read_image(dp_seg_image_dir, format=self.img_format)
            sem_seg_gt = transforms.apply_segmentation(dp_seg_image)
            sem_seg_gt = sem_seg_gt / 255.
            # sem_seg_gt = np.repeat(sem_seg_gt[None,:,:], len(instances._fields['gt_boxes']), 0)
            sem_seg_gt = torch.as_tensor(sem_seg_gt)
            # instances.gt_dp_segms = sem_seg_gt
            dataset_dict["sem_seg"] = sem_seg_gt
        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]

        return dataset_dict

    def _transform_densepose(self, annotation, transforms):
        if not self.densepose_on:
            return annotation

        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # 'None' is accepted by the DensePostList data structure.
            annotation["densepose"] = None

        return annotation
