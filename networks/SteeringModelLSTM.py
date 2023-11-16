import torch.nn as nn
import torchvision.models as models

class SteeringModelLSTM(nn.Module):
    def __init__(self):
        super(SteeringModelLSTM, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.lstm = nn.LSTM(input_size=1000, hidden_size=512, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.resnet(x)
        x = x.view(batch_size, timesteps, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    
    