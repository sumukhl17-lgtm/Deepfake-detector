---
license: mit
metrics:
- f1
- accuracy
pipeline_tag: image-classification
---
# 🧠 EfficientNet-B0 Deepfake Detector (FaceForensics++ C23)

## 👤 Author
**Himanshu Kashyap (Xicor9)**  
MSc Advanced Computer Science  
University of Strathclyde, Glasgow

---

## 📘 Overview
This repository contains a fine-tuned **EfficientNet-B0** model for **deepfake detection** trained on the **FaceForensics++ (FF++) C23** dataset. The model classifies face images or frames into:

- **0 → Real**
- **1 → Fake**

The training data includes multiple manipulation types such as DeepFake, FaceSwap, Face2Face, and NeuralTextures. The model is designed for academic and research use.

---

## 📊 Performance

### **Frame-Level Metrics**
| Metric | Score |
|--------|-------|
| **AUC** | 0.933 |
| **AP** | 0.898 |
| **Accuracy** | 0.852 |
| **F1-Score** | 0.843 |

### **Video-Level Metrics**  
(Mean probability across frames)
| Metric | Score |
|--------|-------|
| **AUC** | 0.94+ |
| **Accuracy** | 0.88+ |

---

## 🏗 Model Architecture & Training Setup

- Base model: **EfficientNet-B0 (ImageNet pretrained)**
- Final classification layer: `Linear(in_features, 2)`
- Loss function: **CrossEntropyLoss**
- Optimizer: **AdamW (lr = 3e-4, weight_decay = 1e-4)**
- LR Scheduler: **CosineAnnealingLR (T_max = 10)**
- Mixed Precision (AMP): **Enabled**
- Epochs: 8
- Hardware: Tesla T4 (Google Colab)

---

## 🔧 How to Use This Model

### Load the model
```python
import torch
import torch.nn as nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load state dict from Hugging Face
state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth",
    map_location=device
)

# Rebuild architecture
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(state_dict)
model.to(device)
model.eval()
```

## 📷 Example Inference

```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("frame.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    prob_fake = torch.softmax(logits, dim=1)[0][1].item()

print("Fake Probability:", prob_fake)
```

## 🗄 Training Dataset

This model was trained on the **FaceForensics++ (FF++) C23** dataset.

### Included Video Types
- **Real videos**
- **Manipulated videos**, including:
  - DeepFake  
  - FaceSwap  
  - Face2Face  
  - NeuralTextures  

All videos were converted into frames and resized to **224 × 224** before training.

---

## 💡 Intended Use

This model is designed for:

- Deepfake research  
- Academic coursework and university projects  
- Benchmarking deepfake detection models  
- Dataset experimentation and model comparison  

❗ **Not intended for real-world forensic, biometric, surveillance, or law-enforcement applications.**

---

## 📎 License

This model is released **strictly for research and educational purposes only**.  
Users must ensure compliance with dataset and model licenses before use.

---

## ❤️ Acknowledgements

- **FaceForensics++** dataset authors  
- **PyTorch / TorchVision** development teams  
- **EfficientNet** authors  
- **Google Colab** for compute resources