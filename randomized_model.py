import torch
import torchvision
from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator, SSDHead


def initialize_weights(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(module.weight, a=0.01)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.uniform_(module.weight)
        torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Sequential):
        for m in module.children():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)


def create_randomized_model(num_classes=2, size=300, nms=0.45):
    model_backbone = torchvision.models.mobilenet_v3_large()
    model_backbone.apply(initialize_weights)
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

# Example usage:
# model = create_randomized_model()
