# 导入操作系统接口模块，用于与操作系统进行交互
import os

# 设置环境变量 "HF_ENDPOINT" 为指定的URL
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置计算设备为第六块GPU（索引从0开始）
device = "cuda:5"

# 从diffusers库导入StableDiffusionPipeline，用于加载和使用稳定扩散模型
from diffusers import StableDiffusionPipeline

# 从自定义采样器模块导入DDIMScheduler（注意：可能与上面导入的DDIMScheduler重复）
from sampler.ddim_scheduling import DDIMScheduler

# 从自定义模型模块导入AutoencoderKL，用于潜在空间的编码和解码
from models.autoencoder_kl import AutoencoderKL

# 从自定义模型模块导入UNet2DConditionModel，用于条件UNet模型
from models.unet_2d_condition import UNet2DConditionModel

# 模型配置
# 设置CLIP文本模型的路径
clip_text_path = "openai/clip-vit-base-patch16"

# 设置稳定扩散预训练模型的路径
pretrained_model = "CompVis/stable-diffusion-v1-4"

# LDM（潜在扩散模型）配置
# 设置DDIM反演的步数为50
NUM_DDIM_STEPS = 50

# 设置是否使用低资源模式，默认为False
LOW_RESOURCE = False

# 设置文本输入的最大词数为77
MAX_NUM_WORDS = 77

# 设置引导尺度，用于控制生成图像的指导强度
GUIDANCE_SCALE = 7.5


def load_pipe():
    # 从预训练模型加载稳定扩散管道，并将其移动到指定的计算设备（GPU）
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model).to(device)

    # 使用DDIM调度器替换现有调度器
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 从预训练模型加载条件UNet模型，并将其移动到指定设备
    pipe.unet = UNet2DConditionModel.from_pretrained(
        pretrained_model, subfolder="unet"
    ).to(device)

    # 从预训练模型加载自动编码器（VAE），并将其移动到指定设备
    pipe.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(
        device
    )
    # 启用VAE的平铺模式，以处理大尺寸图像
    pipe.enable_vae_tiling()

    # 启用VAE的切片模式，以减少内存占用
    pipe.enable_vae_slicing()

    return pipe
