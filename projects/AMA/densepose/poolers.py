import torch
import torch.nn as nn
import math
from detectron2.layers import ROIAlign
from torchvision.ops import RoIPool
from detectron2.modeling.poolers import convert_boxes_to_pooler_format

class MultiROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index on which a canonically-sized box
                should be placed. The default is defined as level 4 in the FPN paper.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) > 0
        # assert isinstance(output_size[0], tuple) and isinstance(output_size[0][0], int)
        self.output_size = output_size
        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    out_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for idx, (out_size, scale) in enumerate(zip(output_size, scales))
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    out_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for idx, (out_size, scale) in enumerate(zip(output_size, scales))
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(out_size, spatial_scale=scale) for idx, (out_size, scale) in enumerate(zip(output_size, scales))
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level))
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert 0 < self.min_level and self.min_level <= self.max_level
        assert self.min_level <= canonical_level and canonical_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x, box_lists):
        """
        Args:
            x (list[Tensor]): A list of feature maps with scales matching those used to
                construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

        Returns:
            A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        output = []
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            output.append(pooler(x_level, pooler_fmt_boxes))

        return output