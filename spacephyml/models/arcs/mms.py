from torch import nn

class PCBaseline_arc(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
                nn.Conv3d(1, 32,
                          kernel_size = (5,3,5), stride = (2,1,2),
                          padding = 0),
                nn.Conv3d(32, 32,
                          kernel_size = (3,3,3), stride = (1,1,1),
                          padding = 0),
                nn.MaxPool3d(2),
                nn.Flatten(),
                nn.Linear(6912, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Softmax(dim = 1))

    def forward(self, x):
        return self.classifier(x)

class PCReduced_arc(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
                nn.Conv3d(1, 1,
                          kernel_size=(5, 3, 5),
                          stride=(2, 1, 2),
                          padding=0),
                nn.MaxPool3d(2),
                nn.Flatten(),
                nn.Linear(343, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Softmax(dim=1))

    def forward(self, x):
        return self.classifier(x)
