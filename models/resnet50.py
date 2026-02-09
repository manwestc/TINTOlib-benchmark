import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, 
                 weights=None,           # Set False by default for synthetic images
                 task_type="classification",     # "classification" or "regression"
                 num_classes=1,                  # only needed for classification
                 freeze_backbone=False,
                 remove_fc_head=False):
        super(ResNet50, self).__init__()

        # Load a ResNet50 with or without pretrained weights
        self.backbone = models.resnet50(weights=weights)

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Save task type
        self.task_type = task_type.lower()

        # Modify the final fully connected layer
        in_features = self.backbone.fc.in_features

        if self.task_type == "regression":
            self.backbone.fc = nn.Linear(in_features, 1)
        if self.task_type == "binary":
            self.backbone.fc = nn.Linear(in_features, 1)
        elif self.task_type == "multiclass":
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def forward(self, x):
        return self.backbone(x)
