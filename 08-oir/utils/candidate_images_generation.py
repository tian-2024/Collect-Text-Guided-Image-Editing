# 导入类型提示相关模块
from typing import Optional

# 导入torch模块
import torch

# 导入自定义的基础工具模块，并命名为basic_utils
import utils.basic_utils as basic_utils

# 导入tqdm中的tqdm，用于进度条
from tqdm import tqdm


# 装饰器，表示在此函数中不计算梯度
@torch.no_grad()
def candidate_images_generation(
    pipe,  # 稳定扩散管道对象
    origin_prompt,  # 原始提示词
    guided_prompt,  # 引导提示词
    prompt_change,  # 提示词变化列表
    all_latents=[],  # 所有潜在向量列表，默认为空列表
    num_inference_steps: int = 50,  # 推理步数，默认为50
    guidance_scale: Optional[float] = 7.5,  # 引导尺度，默认为7.5
    height=512,  # 生成图像的高度，默认为512
    width=512,  # 生成图像的宽度，默认为512
    save_path="",  # 保存路径，默认为空
):
    """
    生成候选图像的函数，通过稳定扩散管道和提示词变化来生成不同的图像。

    参数:
        pipe: 稳定扩散管道对象，用于生成图像。
        origin_prompt (str): 原始提示词，用于生成初始图像。
        guided_prompt (str): 引导提示词，用于引导图像生成。
        prompt_change (list): 提示词变化列表，用于生成多样化的图像。
        all_latents (list, optional): 所有潜在向量列表，默认为空列表。
        num_inference_steps (int, optional): 推理步数，默认为50。
        guidance_scale (float, optional): 引导尺度，用于控制生成图像的指导强度，默认为7.5。
        height (int, optional): 生成图像的高度，默认为512。
        width (int, optional): 生成图像的宽度，默认为512。
        save_path (str, optional): 图像保存路径，默认为空。

    返回:
        images: 生成的图像列表。
    """
    # 获取原始图像的潜在向量
    origin_image_latent = all_latents[0]

    # 计算批处理大小，这里将推理步数除以2作为批大小
    batch_size = num_inference_steps // 2

    # 将原始和引导提示词传入分词器
    text_inputs = pipe.tokenizer(
        [origin_prompt, guided_prompt],
        padding="max_length",  # 使用最大长度填充
        max_length=pipe.tokenizer.model_max_length,  # 设置最大长度为分词器的最大长度
        truncation=True,  # 启用截断
        return_tensors="pt",  # 返回PyTorch张量
    )  # 对提示词进行分词处理

    # 获取输入ID的最大长度
    max_length = text_inputs.input_ids.shape[-1]

    # 获取文本嵌入，将输入ID转移到设备上并通过文本编码器获取嵌入
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]

    # 重复原始上下文以匹配批处理大小，形状为(batch_size, embedding_dim)
    original_context = text_embeddings[0].unsqueeze(0).repeat(batch_size, 1, 1)

    # 初始化潜在向量以进行并行处理
    origin_image_latent, latents = basic_utils.init_latent_parallel(
        origin_image_latent,  # 原始图像潜在向量
        pipe,  # 稳定扩散管道对象
        height,  # 图像高度
        width,  # 图像宽度
        batch_size,  # 批处理大小
        all_latents,  # 所有潜在向量
        num_inference_steps,  # 推理步数
    )

    # 设置调度器的时间步数
    pipe.scheduler.set_timesteps(num_inference_steps)

    # 构建时间步矩阵，重复batch_size次
    temporal_timesteps = (
        torch.cat([pipe.scheduler.timesteps[-1].unsqueeze(0), pipe.scheduler.timesteps])
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    """
       [[  1, 981, 961,  ...,  41,  21,   1],
        [ 21,   1, 961,  ...,  41,  21,   1],
        [ 41,  21,   1,  ...,  41,  21,   1],
        ...,
        [441, 421, 401,  ...,  41,  21,   1],
        [461, 441, 421,  ...,  41,  21,   1],
        [481, 461, 441,  ...,  41,  21,   1]]
    """  # 示例说明时间步矩阵

    # 遍历时间步的第一维
    for m in range(1, temporal_timesteps.shape[0]):
        # 遍历时间步的第二维
        for n in range(temporal_timesteps.shape[1]):
            if n <= m:
                # 更新时间步值，确保时间步的顺序
                temporal_timesteps[m][n] = pipe.scheduler.timesteps[n - m - 1]

    # 设置文本索引为1，指向引导提示词
    text_idx = 1

    # 合并原始上下文和引导上下文
    context = torch.cat(
        [
            original_context,  # 原始上下文
            text_embeddings[text_idx]
            .unsqueeze(0)
            .repeat(
                num_inference_steps // 2, 1, 1
            ),  # 引导提示词的文本嵌入，重复以匹配推理步数的一半
        ]
    )

    # 初始化编辑后的潜在向量为None
    edited_latents = None

    # 遍历所有推理步，从0到num_inference_steps
    for i in tqdm(range(num_inference_steps + 1)):
        # 获取当前时间步
        t = temporal_timesteps[:, i]

        # 执行扩散步骤，更新潜在向量
        latents = basic_utils.diffusion_step_parallel(
            pipe,  # 稳定扩散管道对象
            latents,  # 当前潜在向量
            context,  # 上下文嵌入
            t,  # 当前时间步
            guidance_scale,  # 引导尺度
            low_resource=False,  # 不使用低资源模式
            use_parallel=True,  # 启用并行处理
        )

        if i < num_inference_steps // 2:
            if edited_latents is None:
                # 初始化编辑后的潜在向量
                edited_latents = latents[i].unsqueeze(0)
            else:
                # 将新的潜在向量连接到编辑后的潜在向量中
                edited_latents = torch.cat([edited_latents, latents[i].unsqueeze(0)])

            # 替换当前潜在向量为all_latents中的对应向量
            latents[i] = all_latents[-1 - i]

    # 反向遍历潜在向量的第一维
    for i in reversed(range(latents.shape[0])):
        # 将潜在向量连接到编辑后的潜在向量中
        edited_latents = torch.cat([edited_latents, latents[i].unsqueeze(0)])

    # 设置第一个潜在向量为原始潜在向量
    edited_latents[0] = all_latents[0]

    # 将潜在向量转换为图像
    images = basic_utils.latent2image(pipe.vae, edited_latents)

    # basic_utils.view_images(images, save_path=os.path.join(save_path, prompt_change[text_idx - 1]) + '/', file_name="0000.png")  # 可视化并保存图像（注释掉）

    # 返回生成的图像
    return images
