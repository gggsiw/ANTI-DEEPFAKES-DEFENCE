#!/usr/bin/env python3
print("""
=========================================================
🛡️ OFFLINE IMAGE HARDENING TOOL (ANTI-DEEPFAKE DEFENSE)
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
from torchvision import models, transforms
from PIL import Image
import random
import os

print("""COMMANDS EXAPLES :-
1. python3 ANTI-DEEPFAKES_DEFENSE.py myphoto.jpg protected.png
2. python3 ANTI-DEEPFAKES_DEFENSE.py myphoto.jpg protected.png --steps 50 --epsilon 0.03 --alpha 0.004
""")
#so guys this is optional: install open_clip beforehand (offline wheel)
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

#so guys this is robust transformations
def random_transform(x):
    if random.random() < 0.7:
        scale = random.choice([460, 480, 500, 520])
        x = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

    if random.random() < 0.5:
        noise = torch.randn_like(x) * 0.01
        x = torch.clamp(x + noise, 0, 1)

    return x

#this section is smoothness
def total_variation(x):
    return torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

#ths is frequency perturbation
def frequency_noise(x):
    fft = torch.fft.fft2(x)
    perturb = torch.randn_like(fft) * 0.01
    fft = fft + perturb
    return torch.real(torch.fft.ifft2(fft))

#this for main protection
def protect_image(input_path, output_path, steps=50, epsilon=0.035, alpha=0.004):
    print("\n🛡️ Starting protection...")

    img = Image.open(input_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    x = preprocess(img).unsqueeze(0).to(device)

    #so guys this is models
    print("⚙️ Loading models...")
    resnet = models.resnet50(pretrained=True).to(device).eval()
    efficientnet = models.efficientnet_b0(pretrained=True).to(device).eval()

    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()

    #this is identity baseline
    with torch.no_grad():
        res_orig = resnet(x).detach()
        eff_orig = efficientnet(x).detach()
        clip_orig = clip_model.encode_image(x).detach()

    x_adv = x.clone().detach()

    print("🚀 Running optimization...")

    for step in range(steps):
        x_adv.requires_grad_(True)

        x_t = random_transform(x_adv)
        x_t = torch.clamp(x_t + 0.1 * frequency_noise(x_t), 0, 1)

        res_out = resnet(x_t)
        eff_out = efficientnet(x_t)
        clip_out = clip_model.encode_image(x_t)


        loss_res = -F.mse_loss(res_out, res_orig)
        loss_eff = -F.mse_loss(eff_out, eff_orig)
        loss_clip = -F.cosine_similarity(clip_out, clip_orig).mean()

        rand_target = torch.randn_like(clip_out)
        loss_scramble = F.mse_loss(clip_out, rand_target)

        loss_tv = total_variation(x_adv)

        loss = (
            loss_res +
            loss_eff +
            loss_clip +
            0.6 * loss_scramble +
            0.08 * loss_tv
        )

        resnet.zero_grad()
        efficientnet.zero_grad()
        clip_model.zero_grad()

        loss.backward()

        #this is PGD update
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

        if step % 5 == 0:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f}")

    #this wil save output
    final_img = transforms.ToPILImage()(x_adv.squeeze(0).cpu())
    final_img.save(output_path, "PNG")

    print(f"\n✅ DONE → Saved to: {output_path}")
    print("🛡️ Image is now harder to use for AI models.\n")


#CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline Anti-Deepfake Image Protection Tool")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.035)
    parser.add_argument("--alpha", type=float, default=0.004)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("❌ Input file not found")
        exit()

    protect_image(
        args.input,
        args.output,
        steps=args.steps,
        epsilon=args.epsilon,
        alpha=args.alpha
    )
