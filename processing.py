import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

def predict(image, model):
    model.eval()
    PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(PIL_image).unsqueeze(0)

    with torch.no_grad():
        output = model.forward(img_tensor)
    output_prob = torch.exp(output).squeeze()

    return output_prob.argmax().item()

def load_model():
    model = models.vgg11()

    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 7),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model.load_state_dict(torch.load("fer_model.pt"))

    return model
