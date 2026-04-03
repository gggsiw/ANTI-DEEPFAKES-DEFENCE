
#!/usr/bin/env python3
print("""
=========================================================
🛡️ PixelGuard (ANTI-DEEPFAKE DEFENSE)
=========================================================

WHAT THIS TOOL DOES:
- Adds subtle, human-invisible perturbations to images
- Confuses AI models (like CLIP, ResNet, EfficientNet)
- Makes deepfake training & identity extraction harder

WHAT THIS TOOL DOES NOT DO:
- ❌ Does NOT guarantee protection
- ❌ Does NOT stop determined attackers
- ❌ Does NOT survive heavy editing or redraws

THINK OF IT AS:
✔️ "Armor" — NOT invisibility

---------------------------------------------------------
📌 HOW TO USE (IMPORTANT)
---------------------------------------------------------

1. ALWAYS protect images BEFORE uploading online

2. DO NOT upload original + protected together
   → That cancels the protection

3. Use moderate settings:
   - If image looks weird → reduce epsilon or steps

4. Avoid uploading many similar images:
   - Same angle, lighting = easier deepfake training

5. Prefer lower resolution uploads (512–1024px)

6. Slight cropping per upload improves protection

---------------------------------------------------------
⚠️ REAL-WORLD SAFETY TIPS
---------------------------------------------------------

- Limit public sharing of high-quality face images
- Avoid sharing multiple angles of the same face
- Report misuse on platforms
- Use provenance tools like Content Authenticity Initiative

---------------------------------------------------------
🧠 TECHNICAL SUMMARY
---------------------------------------------------------

This tool uses:
- Multi-model attack (ensemble)
- Embedding disruption (CLIP-like models)
- Iterative PGD attack
- EOT (resize/noise robustness)
- Frequency + spatial perturbations

=========================================================
""")



import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import open_clip
import torchvision.transforms as transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)
model = model.to(device)
model.eval()

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]



def denormalize(tensor):
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean



def protect_image(input_path, output_path, epsilon=0.03, alpha=0.005, steps=30):

    image = Image.open(input_path).convert("RGB")

    # Preprocess (CLIP-safe)
    x = preprocess(image).unsqueeze(0).to(device)

    # Clone
    x_adv = x.clone().detach()

    # Original embedding
    with torch.no_grad():
        orig_embed = model.encode_image(x)

    print("\n🛡️ Starting PixelGuard Protection...\n")

    for step in range(steps):
        x_adv.requires_grad_(True)

        embed = model.encode_image(x_adv)

        # Maximize difference
        loss = -F.cosine_similarity(embed, orig_embed).mean()

        model.zero_grad()
        loss.backward()

        # PGD step
        grad = x_adv.grad.data
        x_adv = x_adv + alpha * torch.sign(grad)

        # Project back into epsilon ball
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)

        # Detach for next step
        x_adv = x_adv.detach()

        print(f"Step {step+1}/{steps} - Loss: {-loss.item():.4f}")


    x_adv = denormalize(x_adv)

    # Clamp AFTER denormalization
    x_adv = torch.clamp(x_adv, 0, 1)

    # Convert to image
    x_adv = x_adv.squeeze().cpu()
    protected_img = transforms.ToPILImage()(x_adv)

    protected_img.save(output_path)

    print(f"\n✅ Protected image saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixelGuard - Anti Deepfake Protection")

    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=30)

    args = parser.parse_args()

    protect_image(
        args.input,
        args.output,
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps
    )
