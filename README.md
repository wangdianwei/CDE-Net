# 🌊 CDE-Net  
A Cascaded Network Based on Color Correction and Dynamic Details for Underwater Image Enhancement

📄 Paper: *CDE-Net: A Cascaded Network Based on Color Correction and Dynamic Details for Underwater Image Enhancement*  
Any questions, pls contact to: [your_email_here]

---

## 📌 Introduction

CDE-Net is a cascaded deep network for **underwater image enhancement (UIE)**, designed to address three major degradation issues in underwater scenes:

- Color distortion
- Low contrast
- Blurred details

Unlike conventional single-stage enhancement networks, CDE-Net decomposes the enhancement process into two coordinated stages:

1. **Color Correction Stage**
2. **Detail Enhancement Stage**

The first stage performs global and local chromatic correction in the **Lab color space**, while the second stage restores texture details and suppresses scattering artifacts through a **dynamic weight-adjustment convolution mechanism**.

This cascaded design enables CDE-Net to achieve a better balance between **color fidelity**, **local contrast**, and **structural detail recovery**.

---

## 🚀 Key Features

- 🎨 Cascaded Color Correction + Detail Enhancement  
  Decomposes underwater image enhancement into two complementary stages.

- 🌈 Lab-Space Color Correction  
  Corrects the *a* and *b* chrominance channels and enhances the *L* luminance channel via linear stretching.

- 🔍 Dynamic Weight-Adjustment Convolution (DWAC)  
  Adaptively modulates convolution kernels according to local degradation intensity for better texture restoration and noise suppression.

- 📉 Hybrid Optimization Losses  
  Combines:
  - Discrete Cosine Distance (DCD) Loss for chromatic consistency
  - Contrast Energy Loss for local contrast and edge preservation

- ⚖️ Strong Quantitative and Visual Performance  
  Achieves competitive results on public underwater image enhancement benchmarks.

---

## 🏗️ Framework Overview

The CDE-Net pipeline consists of two stages:

### 1. Color Correction Stage (CD-Net)
- Convert the input underwater image from RGB to **Lab color space**
- Feed the **a** and **b** channels into CD-Net for chromatic correction
- Apply **1% / 99% quantile linear stretching** to the **L** channel
- Fuse corrected Lab channels and convert back to RGB

### 2. Detail Enhancement Stage (DE-Net)
- Use an encoder–decoder framework to refine structural details
- Introduce **DWAC** to dynamically enhance textures and edges
- Suppress scattering noise while preserving natural appearance

Overall, the cascaded design enables the network to first fix global color bias and then recover local structures more effectively.

---

## 🧠 Core Innovations

### 👻 Dynamic Weight-Adjustment Convolution (DWAC)
DWAC is the key component in the detail enhancement stage.  
It generates **input-conditioned convolution weights** based on spatial degradation statistics, allowing the network to:

- Enhance degraded texture regions
- Recover fine edges
- Suppress noise and scattering artifacts

### 📏 Discrete Cosine Distance Loss
A Lab-space color correction loss that constrains the chromaticity direction of enhanced images, improving color fidelity and reducing color cast.

### ✨ Contrast Energy Loss
A gradient-based contrast supervision term that encourages sharper structures and stronger local detail restoration.

---

## 📊 Experimental Results

### Quantitative Results on LSUI

| Method   | PSNR | SSIM | UCIQE | UIQM | MOS |
|----------|------|------|-------|------|-----|
| CDE-Net  | 26.33 | 0.946 | 0.599 | 1.111 | 4.3 |

### Quantitative Results on UIEB

| Method   | PSNR | SSIM | UCIQE | UIQM | MOS |
|----------|------|------|-------|------|-----|
| CDE-Net  | 20.41 | 0.871 | 0.628 | 0.866 | 4.0 |

✔ Ranked first on multiple evaluation metrics  
✔ Produces better visual quality with more natural colors and clearer details  
✔ Achieves a strong balance between restoration quality and model efficiency  

---

## 🗂️ Datasets

The framework is evaluated on the following public underwater image enhancement datasets:

- **LSUI**
- **UIEB**

### Evaluation Metrics
The model is assessed using:

- **PSNR**
- **SSIM**
- **UCIQE**
- **UIQM**
- **MOS** (Mean Opinion Score)

---

## ⚙️ Training Details

- Framework: **PyTorch**
- Optimizer: **Adam**
- Initial learning rate: **1e-4**
- Batch size: **8**
- Training epochs: **200**
- Input patch size: **256 × 256**
- Data augmentation:
  - Random cropping
  - Random rotation
  - Random flipping

---

## 📈 Ablation Study

Ablation experiments show that the following components all contribute to performance improvement:

- Color correction loss
- Contrast energy loss
- DWAC module

The full CDE-Net achieves the best overall performance among all ablation variants.

---

## 🔬 Visual Performance

Compared with existing underwater image enhancement methods, CDE-Net:

- Reduces blue-green color bias more effectively
- Preserves more local textures and object boundaries
- Produces more natural contrast
- Avoids oversaturation and excessive artifacts

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{cdenet,
  title={CDE-Net: A Cascaded Network Based on Color Correction and Dynamic Details for Underwater Image Enhancement},
  author={Wang, Dianwei and Yang, Yi and Fang, Jie and Li, Yuanqing and Ai, Da and Tang, Jianing and Huang, Loulin},
  journal={Under review},
  year={2025}
}
