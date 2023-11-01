import torch.nn as nn
import torchvision.models as models

# Define the model
class SteeringModel(nn.Module):
    def __init__(self):
        super(SteeringModel, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Normalize the input image
        # x = (x / 127.5) - 1.0
        x = self.resnet(x)
        return x