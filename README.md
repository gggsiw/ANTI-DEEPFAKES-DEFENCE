# 🛡️ Anti-Deepfake Defense Tool

An offline image hardening tool designed to make deepfake generation and AI-based identity extraction more difficult.

## 🚀 Features
- Fully offline (no data upload)
- Multi-model adversarial protection (ResNet + EfficientNet + CLIP-like)
- Iterative PGD-based perturbation
- Robust against resizing and noise (EOT)
- Frequency + spatial domain perturbations
- Minimal visual distortion

## ⚠️ Important Disclaimer
This tool does NOT guarantee protection against deepfakes.

It is designed to:
- Increase the difficulty of misuse
- Reduce model reliability
- Add a defensive layer

Advanced attackers and custom-trained models may still bypass these protections.

## 🔐 Privacy
- No images are uploaded or transmitted
- All processing happens locally
- No data is stored or logged

## 🧠 How it Works
The tool applies adversarial perturbations to images by disrupting:
- Feature representations in models like :contentReference[oaicite:0]{index=0}  
- Classification embeddings in CNN architectures  

This reduces identity consistency across AI systems.


## HAVE IDEA TO MAKE IT ADVANCE?? YOU CAN MODIFY AND UPLOAD IT AS YOURS, I DON'T DO COPYRIGHT, SO LET'S TAKE A STEP TO STAND AGAINST DEEPFAKES.


## 📌 Usage

# Clone repo
git clone https://github.com/yourusername/PixelGuard

# Install dependencies
pip install -r requirements.txt

# Run
python3 pixelguard.py myphoto.jpg protected.png

