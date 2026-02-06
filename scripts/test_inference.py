import torch
from torchvision import transforms
from PIL import Image
import sys
sys.path.append('/content/solar-defect-engine')
from core.model import SolarCNN

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarCNN(num_classes=6).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        idx = torch.argmax(output, 1).item()
    classes = ['Bird-drop', 'Crack', 'Dusty', 'Electrical', 'Normal', 'Physical-Damage']
    print(f"Result: {classes[idx]}")

if __name__ == "__main__":
    print("Testing pipeline logic...")
