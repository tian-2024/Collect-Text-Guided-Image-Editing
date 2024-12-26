# [Collect Text-Guided Image Editing (TIE) Methods](https://github.com/tian-2024/TIE-Collect)


Each directory contains the raw code for text-guided image editing methods using diffusion models.

The main goal is to ensure the code runs successfully in a consistent conda environment. This may require modifying some parts of the code to ensure the entire project operates within the same environment.

## install environment

The cuda environment used here is 12.1.

### option 1

```bash
conda create -n tie python=3.10
conda activate tie
pip install -r requirements.txt
```

### option 2

manually install the following packages:

```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install diffusers
pip install torchvision
pip install transformers
pip install accelerate
pip install opencv-python
pip install ipywidgets
pip install einops
pip install matplotlib
pip install pytorch-lightning
pip install omegaconf
```

## Implemented Methods

### Synthetic Image Editing


- [x] 01-ptp:  (2023 ICLR) [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://prompt-to-prompt.github.io/)

### Real Image Editing

- [x] 02-nti:  (2023 CVPR) [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://null-text-inversion.github.io/)
- [x] 03-dds:  (2023 ICCV) [Delta Denoising Score](https://delta-denoising-score.github.io/)
- [x] 04-fpe:  (2024 CVPR) [Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing](https://github.com/alibaba/EasyNLP/tree/master/diffusion/FreePromptEditing)
- [x] 05-cds:  (2024 CVPR) [Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing](https://github.com/HyelinNAM/ContrastiveDenoisingScore)
- [x] 06-masactrl:  (2023 ICCV) [MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://ljzycmd.github.io/projects/MasaCtrl/)
- [x] 07-sega:  (2023 NeurIPS) [SEGA: Instructing Text-to-Image Models using Semantic Guidance](https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion)
- [x] 08-oir:  (2024 ICLR) [Object-aware Inversion and Reassembly for Image Editing](https://aim-uofa.github.io/OIR-Diffusion/)

## Planned Methods

1. [Paint by Example: Exemplar-based Image Editing with Diffusion Models](https://github.com/Fantasy-Studio/Paint-by-Example)
2. [Imagic: Text-Based Real Image Editing with Diffusion Models](https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic)