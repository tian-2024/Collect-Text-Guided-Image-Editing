# 导入类型提示相关的模块
from typing import Optional

# 从tqdm库导入tqdm，用于显示循环进度条
from tqdm import tqdm

# 导入PyTorch库
import torch

# 导入自定义的basic_utils模块，并命名为basic_utils
import utils.basic_utils as basic_utils


def change_all_masks_shape(mask, latent):
    """
    调整所有mask的形状以匹配潜在表示（latent）的尺寸。

    参数:
        mask (torch.Tensor): 输入的mask，形状为(N, H, W)。
        latent (torch.Tensor): 潜在表示，形状为(N, C, H, W)。

    返回:
        torch.Tensor: 调整后的mask，形状为(N, C, H, W)。
    """
    # 增加一个通道维度并将mask移动到与latent相同的设备上
    mask = mask.unsqueeze(1).to(latent.device)

    # 将mask转换为浮点类型，并调整大小以匹配latent的高度和宽度，使用最近邻插值方式
    mask = (
        torch.nn.functional.interpolate(
            mask.float(),
            size=(latent.shape[2], latent.shape[3]),
            mode="nearest",
        )
        > 0.5
    )

    # 重复mask以匹配latent的通道数，并转换为整型
    mask = mask.repeat(1, latent.shape[1], 1, 1).int()

    # 返回调整后的mask
    return mask


# 装饰器，表示在函数执行过程中不计算梯度
@torch.no_grad()
def oir(
    model,
    prompts,
    optimal_inversion_steps,
    num_inference_steps: int = 50,  # 推理的步数，默认为50
    guidance_scale: Optional[float] = 7.5,  # 指导尺度，默认为7.5
    x_t: Optional[torch.FloatTensor] = None,  # 可选的初始潜在表示
    return_type="image",  # 返回类型，默认为'image'，也可以是'latent'
    all_latents=None,  # 所有潜在表示的字典
    all_masks=None,  # 所有mask的字典
    ddim_inversion=None,  # DDIM反演对象
    reinversion_steps=0,  # 反演步数，默认为0
    prompt_changes=[],  # 提示变化的列表
    reassembly_step=0,  # 重新组装的步数，默认为0
    height=512,  # 生成图像的高度，默认为512
    width=512,  # 生成图像的宽度，默认为512
):
    """
    执行Optimal Inversion Refinement (OIR)过程，以生成编辑后的图像或潜在表示。

    参数:
        model: 使用的模型，包含tokenizer、text_encoder、scheduler等组件。
        prompts (list): 提示词列表，包含原始提示、引导提示和目标提示。
        optimal_inversion_steps (dict): 每个提示变化对应的最佳反演步数。
        num_inference_steps (int): 推理的总步数。
        guidance_scale (float, optional): 指导尺度，用于调整生成图像的指导强度。
        x_t (torch.FloatTensor, optional): 初始的潜在表示，如果提供则使用该值进行初始化。
        return_type (str): 返回类型，可以是'image'或'latent'。
        all_latents (dict, optional): 所有潜在表示的字典。
        all_masks (dict, optional): 所有mask的字典。
        ddim_inversion: DDIM反演对象，用于反演潜在表示。
        reinversion_steps (int): 反演的步数。
        prompt_changes (list): 提示变化的列表。
        reassembly_step (int): 重新组装的步数。
        height (int): 生成图像的高度。
        width (int): 生成图像的宽度。

    返回:
        image (PIL.Image 或 torch.Tensor): 生成的图像或潜在表示。
        reassembly_latent (torch.Tensor): 重新组装后的潜在表示。
    """
    # 获取提示词的数量，作为批次大小
    batch_size = len(prompts)

    # 初始化潜在表示
    latent, latents = basic_utils.init_latent(x_t, model, height, width, batch_size)

    # 设置调度器的时间步数
    model.scheduler.set_timesteps(num_inference_steps)

    # 初始化所有潜在mask的字典
    all_latent_masks = {}
    for key in all_masks.keys():
        # 调整每个mask的形状以匹配潜在表示
        all_latent_masks[key] = change_all_masks_shape(all_masks[key], latents)

    # 分离原始提示、引导提示和目标提示
    origin_prompt, guided_prompts, target_prompt = (
        prompts[0],
        prompts[1:-1],
        prompts[-1],
    )

    # 初始化所有嵌入的字典
    all_embeddings = {}
    for prompt_change, guided_prompt in zip(prompt_changes, guided_prompts):
        # 对原始提示和引导提示进行分词
        all_embeddings[prompt_change] = model.tokenizer(
            [origin_prompt, guided_prompt],
            padding="max_length",  # 填充到最大长度
            max_length=model.tokenizer.model_max_length,  # 设置最大长度
            truncation=True,  # 启用截断
            return_tensors="pt",  # 返回PyTorch张量
        )

        # 获取文本编码的嵌入向量
        all_embeddings[prompt_change] = model.text_encoder(
            all_embeddings[prompt_change].input_ids.to(model.device)
        )[0]

    # 对原始提示和目标提示进行分词并获取嵌入向量
    target_embedding = model.tokenizer(
        [origin_prompt, target_prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    target_embedding = model.text_encoder(target_embedding.input_ids.to(model.device))[
        0
    ]

    # 对原始提示进行分词并获取嵌入向量
    origin_embedding = model.tokenizer(
        [origin_prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    origin_embedding = model.text_encoder(origin_embedding.input_ids.to(model.device))[
        0
    ]

    # 初始化引导潜在表示的字典
    guided_latents = {}
    for prompt_change in prompt_changes:
        # 获取当前提示变化的上下文嵌入
        context = all_embeddings[prompt_change]

        # 获取最佳反演步数对应的潜在表示
        latent = all_latents[optimal_inversion_steps[prompt_change]]

        # 获取相关的时间步数
        timesteps = model.scheduler.timesteps[-optimal_inversion_steps[prompt_change] :]

        # 计算重新组装的停止步数
        stop_for_reassembly = len(timesteps) - reassembly_step

        # 遍历时间步数，并显示进度条
        for i, t in enumerate(tqdm(timesteps)):
            # 执行扩散步骤
            latent = basic_utils.diffusion_step(
                model,
                latent,
                context,
                t,
                guidance_scale,
                low_resource=False,
            )

            if i == stop_for_reassembly - 1:
                # 保存引导后的潜在表示
                guided_latents[prompt_change] = latent
                break  # 退出循环

    # 裁剪编辑区域和非编辑区域，并使用它们构建重新组装的潜在表示
    reassembly_latent = (
        all_latent_masks["non_editing_region_mask"] * all_latents[reassembly_step]
    )

    for prompt_change in prompt_changes:
        # 将各个编辑区域的潜在表示相加
        reassembly_latent += (
            all_latent_masks[prompt_change] * guided_latents[prompt_change]
        )

    # 使用重新反演和更改提示到目标提示进行引导的去噪过程
    reassembly_latent = reinversion_and_denoise(
        model,
        reassembly_latent,
        target_embedding,
        origin_embedding,
        ddim_inversion,
        reinversion_steps,
        reassembly_step,
        guidance_scale,
    )

    # 根据返回类型转换潜在表示为图像或返回潜在表示
    image = (
        basic_utils.latent2image(model.vae, reassembly_latent)
        if return_type == "image"
        else latents
    )

    # 可视化并保存图像（注释掉）
    # basic_utils.view_images(images, save_path=os.path.join(save_path, prompt_change[text_idx - 1]) + '/', file_name="0000.png")

    # 返回生成的图像和重新组装的潜在表示
    return image, reassembly_latent


def reinversion_and_denoise(
    model,
    reassembly_latent,
    target_embedding,
    origin_embedding,
    ddim_inversion,
    reinversion_steps,
    reassembly_step,
    guidance_scale,
):
    """
    执行重新反演和引导的去噪过程，以生成最终的潜在表示。

    参数:
        model: 使用的模型，包含scheduler等组件。
        reassembly_latent (torch.Tensor): 重新组装的潜在表示。
        target_embedding (torch.Tensor): 目标提示的嵌入向量。
        origin_embedding (torch.Tensor): 原始提示的嵌入向量。
        ddim_inversion: DDIM反演对象，用于反演潜在表示。
        reinversion_steps (int): 反演的步数。
        reassembly_step (int): 重新组装的步数。
        guidance_scale (float): 指导尺度，用于调整生成图像的指导强度。

    返回:
        reassembly_latent (torch.Tensor): 经过重新反演和去噪后的潜在表示。
    """
    # 执行DDIM反演
    reassembly_latent = ddim_inversion.reinversion(
        reassembly_latent,
        origin_embedding,
        reassembly_step,
        reinversion_steps,
    )

    # 遍历相关的时间步数
    for t in model.scheduler.timesteps[-(reinversion_steps + reassembly_step) :]:
        # 执行扩散步骤进行去噪
        reassembly_latent = basic_utils.diffusion_step(
            model,
            reassembly_latent,
            target_embedding,
            t,
            guidance_scale,
            low_resource=False,
        )

    # 返回处理后的潜在表示
    return reassembly_latent
