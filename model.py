import torchvision
from torchvision.models.detection.ssd import (SSD, DefaultBoxGenerator, SSDHead)


def create_model(num_classes=2, size=300, nms=0.45):
    model_backbone = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    )

    # Remove the last classification layer from MobileNetV3
    backbone = model_backbone.features

    out_channels = [960, 960, 960, 960, 960, 960]

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    )
    num_anchors = anchor_generator.num_anchors_per_location()

    head = SSDHead(out_channels, num_anchors, num_classes)

    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(size, size),
        head=head,
        nms_thresh=nms
    )

    return model
