# [Collect Text-Guided Image Editing Methods](https://github.com/tian-2024/TIE-Collect)

| Method      | Year | Conference | Project                                                                                                                                                                        |
| ----------- | ---- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 01-ptp      | 2023 | CVPR       | [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://prompt-to-prompt.github.io/)                                                                             |
| 02-nti      | 2023 | CVPR       | [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://null-text-inversion.github.io/)                                                            |
| 03-dds      | 2023 | ICCV       | [Delta Denoising Score](https://delta-denoising-score.github.io/)                                                                                                              |
| 04-fpe      | 2024 | CVPR       | [Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing](https://github.com/alibaba/EasyNLP/tree/master/diffusion/FreePromptEditing) |
| 05-cds      | 2024 | CVPR       | [Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing](https://github.com/HyelinNAM/ContrastiveDenoisingScore)                                           |
| 06-masactrl | 2023 | ICCV       | [MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://ljzycmd.github.io/projects/MasaCtrl/)                                 |
| 07-sega     | 2023 | NeurIPS    | [SEGA: Instructing Text-to-Image Models using Semantic Guidance](https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion)                                |
| 08-oir      | 2024 | ICLR       | [Object-aware Inversion and Reassembly for Image Editing](https://aim-uofa.github.io/OIR-Diffusion/)                                                                           |
| 09-leditspp | 2024 | CVPR       | [LEDITS++: Limitless Image Editing using Text-to-Image Models](https://leditsplusplus-project.static.hf.space/index.html)                                                      |

- Each directory contains the raw code for text-guided image editing methods using diffusion models.
- The main goal is to ensure the code runs smoothly in a consistent conda environment with the latest version of diffusers.
- This may require modifying parts of the code to ensure compatibility within the same environment.


## environment setup

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