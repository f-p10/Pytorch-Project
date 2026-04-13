import torch
from torchvision import models, transforms
from pathlib import Path

current_path = Path.cwd()
target_path = current_path.parent / "training" / "model.pth"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_model():
    model = models.resnet50(pretrained=True)
    num_inputs = model.fc.in_features
    model.fc = torch.nn.Linear(num_inputs, 6)

    model.load_state_dict(torch.load(target_path, map_location='cpu'))
    model.eval()
    return model

def classify_image(model, image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()