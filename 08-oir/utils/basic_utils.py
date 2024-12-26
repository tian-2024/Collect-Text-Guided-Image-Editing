# 导入NumPy库，用于数组和矩阵运算
import numpy as np

# 从tqdm库导入trange，用于显示循环进度条
from tqdm import trange

# 导入PyTorch库，用于深度学习
import torch

# 导入OpenCV库，用于图像处理
import cv2

# 导入操作系统接口模块，用于与操作系统进行交互
import os

# 导入时间模块，用于时间相关操作

# 从tqdm库导入用于Jupyter Notebook的进度条

# 导入Matplotlib的pyplot模块，用于绘图

# 从IPython.display导入display，用于在Notebook中显示图像

# 从PIL库导入Image、ImageDraw和ImageFont模块，用于图像处理和绘制
from PIL import Image

# 导入类型提示相关模块
from typing import Tuple

# 从配置模块导入相关函数和变量
from configs.basic_config import (
    load_pipe,  # 稳定扩散管道对象
    NUM_DDIM_STEPS,  # DDIM反演的步数
    GUIDANCE_SCALE,  # 引导尺度，用于控制生成图像的指导强度
    clip_text_path,  # CLIP模型的路径
)

# 从自定义工具模块导入生成候选图像的函数

# 从自定义工具模块导入选择最优候选图像的函数

# 从自定义采样器模块导入DDIM反演类


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
):
    """
    在图像下方添加文本的函数。

    参数:
        image (np.ndarray): 输入的图像，形状为(H, W, C)。
        text (str): 要添加的文本。
        text_color (Tuple[int, int, int], optional): 文本颜色，默认为黑色(0, 0, 0)。

    返回:
        np.ndarray: 添加文本后的新图像，形状为(H + offset, W, C)。
    """
    # 获取图像的高度、宽度和通道数
    h, w, c = image.shape

    # 计算下方文本区域的高度偏移量（高度的20%）
    offset = int(h * 0.2)

    # 创建一个白色背景的新图像，尺寸为原图高度加上偏移量
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255

    # 设置字体类型为Hershey Simplex
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 将原始图像放置在新图像的上方
    img[:h] = image

    # 获取文本的尺寸（宽度和高度）
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # 计算文本的水平起始位置，使其居中
    text_x = (w - textsize[0]) // 2

    # 计算文本的垂直起始位置，位于偏移区域中间
    text_y = h + offset - textsize[1] // 2

    # 在图像上绘制文本
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)

    # 返回带有文本的新图像
    return img


def view_images(images, save_path, file_name, num_rows=1, offset_ratio=0.02):
    """
    显示并保存多张图像的函数，将图像排列成指定的行数，并在图像之间添加间隔。

    参数:
        images (list或np.ndarray): 要显示的图像列表或4维张量。
        save_path (str): 图像保存的目录路径。
        file_name (str): 保存的文件名。
        num_rows (int, optional): 图像排列的行数，默认为1。
        offset_ratio (float, optional): 图像之间的间隔比例，默认为0.02。

    返回:
        None
    """
    # 判断images的类型并计算需要填充的空白图像数量
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    # 创建一个白色空白图像用于填充
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255

    # 将所有图像转换为uint8类型，并添加空白图像
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty

    # 获取总图像数量
    num_items = len(images)

    # 获取单张图像的高度、宽度和通道数
    h, w, c = images[0].shape

    # 计算图像之间的间隔像素数
    offset = int(h * offset_ratio)

    # 计算每行的图像数量
    num_cols = num_items // num_rows

    # 创建一个用于拼接的白色背景图像
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )

    # 将每张图像粘贴到拼接图像的指定位置
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    # 将NumPy数组转换为PIL图像
    pil_img = Image.fromarray(image_)

    # 如果保存路径不存在，创建目录
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # 保存拼接后的图像到指定路径
    pil_img.save(os.path.join(save_path, file_name))


def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    """
    执行一次扩散步骤，用于更新潜在表示。

    参数:
        model: 使用的模型，包含unet和scheduler等组件。
        latents (torch.Tensor): 当前的潜在表示。
        context (torch.Tensor): 上下文嵌入，用于条件控制。
        t (torch.Tensor): 当前时间步。
        guidance_scale (float): 引导尺度，用于控制生成图像的指导强度。
        low_resource (bool, optional): 是否使用低资源模式，默认为False。

    返回:
        torch.Tensor: 更新后的潜在表示。
    """
    if low_resource:
        # 在低资源模式下，仅使用少量资源进行预测
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]  # 无条件噪声预测
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )[
            "sample"
        ]  # 有条件噪声预测
    else:
        # 将潜在向量复制一份，用于有条件和无条件预测
        latents_input = torch.cat([latents] * 2)

        # 获取噪声预测
        noise_pred = model.unet(
            latents_input,
            t,
            encoder_hidden_states=context,
        )["sample"]

        # 将噪声预测分离为无条件和有条件部分
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    # 结合噪声预测，应用指导尺度
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )

    # 使用调度器更新潜在表示
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]

    # 返回更新后的潜在表示
    return latents


def diffusion_step_parallel(
    pipe, latents, context, t, guidance_scale, low_resource=False, use_parallel=True
):
    """
    并行执行一次扩散步骤，用于更新潜在表示。

    参数:
        pipe: 使用的管道，包含unet和scheduler等组件。
        latents (torch.Tensor): 当前的潜在表示。
        context (torch.Tensor): 上下文嵌入，用于条件控制。
        t (torch.Tensor): 当前时间步。
        guidance_scale (float): 引导尺度，用于控制生成图像的指导强度。
        low_resource (bool, optional): 是否使用低资源模式，默认为False。
        use_parallel (bool, optional): 是否使用并行处理，默认为True。

    返回:
        torch.Tensor: 更新后的潜在表示。
    """
    if low_resource:
        # 在低资源模式下，仅使用少量资源进行预测
        noise_pred_uncond = pipe.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]  # 无条件噪声预测
        noise_prediction_text = pipe.unet(latents, t, encoder_hidden_states=context[1])[
            "sample"
        ]  # 有条件噪声预测
    else:
        # 将潜在向量复制一份，用于有条件和无条件预测
        latents_input = torch.cat([latents] * 2)

        # 拼接时间步信息，用于并行预测
        noise_pred = pipe.unet(
            latents_input,
            torch.cat([t, t]),
            encoder_hidden_states=context,
            use_parallel=use_parallel,  # 是否使用并行处理
        )["sample"]

        # 将噪声预测分离为无条件和有条件部分
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    # 结合噪声预测，应用指导尺度
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )

    # 使用调度器更新潜在表示
    latents = pipe.scheduler.step(noise_pred, t, latents, use_parallel=True)[
        "prev_sample"
    ]

    # 返回更新后的潜在表示
    return latents


def latent2image(vae, latents):
    """
    将潜在表示转换为图像。

    参数:
        vae: 变分自编码器模型，用于解码潜在表示。
        latents (torch.Tensor): 潜在表示。

    返回:
        np.ndarray: 生成的图像，形状为(H, W, C)。
    """
    # 反标准化潜在表示
    latents = 1 / 0.18215 * latents

    # 使用VAE解码潜在表示，生成图像
    image = vae.decode(latents)["sample"]

    # 将图像值缩放到[0, 1]范围内
    image = (image / 2 + 0.5).clamp(0, 1)

    # 将图像转换为NumPy数组，并调整维度顺序为(H, W, C)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()

    # 将图像值缩放到[0, 255]并转换为无符号8位整数
    image = (image * 255).astype(np.uint8)

    # 返回生成的图像
    return image


def image2latent(vae, image):
    """
    将图像转换为潜在表示。

    参数:
        vae: 变分自编码器模型，用于编码图像。
        image (PIL.Image或np.ndarray或torch.Tensor): 输入的图像。

    返回:
        torch.Tensor: 潜在表示。
    """
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        if isinstance(image, Image.Image):
            # 如果图像是PIL Image，转换为NumPy数组并取前三个通道
            image = np.array(image)[:, :, :3]

        if isinstance(image, torch.Tensor) and image.dim() == 4:
            # 如果图像已经是4维张量，直接使用
            latents = image
        else:
            # 将图像归一化到[-1, 1]范围
            image = torch.from_numpy(image).float() / 127.5 - 1

            # 调整维度顺序为(C, H, W)，添加批次维度，并移动到VAE所在设备
            image = image.permute(2, 0, 1).unsqueeze(0).to(vae.device)

            # 使用VAE编码图像，获取潜在分布的均值
            latents = vae.encode(image)["latent_dist"].mean

            # 标准化潜在向量
            latents = latents * 0.18215

    # 返回潜在表示
    return latents


def init_latent(latent, model, height, width, batch_size):
    """
    初始化潜在表示，如果未提供则随机生成，并扩展到批次大小。

    参数:
        latent (torch.Tensor或None): 输入的潜在表示，如果为None则随机初始化。
        model: 使用的模型，包含unet等组件。
        height (int): 生成图像的高度。
        width (int): 生成图像的宽度。
        batch_size (int): 批次大小。

    返回:
        tuple: (原始潜在表示, 扩展后的潜在表示)
    """
    if latent is None:
        # 如果未提供潜在表示，随机初始化一个潜在向量
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // 8, width // 8)
        )

    # 扩展潜在向量以匹配批次大小，并移动到模型所在设备
    latents = latent.expand(
        batch_size, model.unet.config.in_channels, height // 8, width // 8
    ).to(model.device)

    # 返回原始潜在向量和扩展后的潜在向量
    return latent, latents


def init_latent_parallel(
    latent, model, height, width, batch_size, all_latents, num_ddim_steps
):
    """
    初始化并行潜在表示，将多个潜在向量串联起来。

    参数:
        latent (torch.Tensor): 输入的潜在表示。
        model: 使用的模型，包含unet等组件。
        height (int): 生成图像的高度。
        width (int): 生成图像的宽度。
        batch_size (int): 批次大小。
        all_latents (list): 所有潜在向量的列表。
        num_ddim_steps (int): DDIM步数。

    返回:
        tuple: (原始潜在表示, 串联后的潜在表示)
    """
    # 获取所有潜在向量的第二个元素
    latents = all_latents[1]

    # 将潜在向量串联起来
    for i in range(1, num_ddim_steps // 2):
        latents = torch.cat([latents, all_latents[i + 1]])

    # 返回原始潜在向量和串联后的潜在向量
    return latent, latents


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    """
    加载并裁剪图像到512x512像素。

    参数:
        image_path (str或np.ndarray): 图像的路径或NumPy数组。
        left (int, optional): 左侧裁剪像素数，默认为0。
        right (int, optional): 右侧裁剪像素数，默认为0。
        top (int, optional): 顶部裁剪像素数，默认为0。
        bottom (int, optional): 底部裁剪像素数，默认为0。

    返回:
        np.ndarray: 裁剪并调整大小后的图像，形状为(512, 512, 3)。
    """
    if isinstance(image_path, str):
        # 如果路径是字符串，加载图像并取前三个通道
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        # 否则直接使用传入的图像
        image = image_path

    # 获取图像的高度、宽度和通道数
    h, w, c = image.shape

    # 确保左边裁剪不超过图像宽度
    left = min(left, w - 1)

    # 确保右边裁剪不超过图像宽度
    right = min(right, w - left - 1)

    # 确保顶部裁剪不超过图像高度
    top = min(top, h - top - 1)

    # 确保底部裁剪不超过图像高度
    bottom = min(bottom, h - top - 1)

    # 裁剪图像
    image = image[top : h - bottom, left : w - right]

    # 获取裁剪后图像的尺寸
    h, w, c = image.shape

    if h < w:
        # 计算水平偏移量以使图像方正
        offset = (w - h) // 2

        # 裁剪宽度以匹配高度
        image = image[:, offset : offset + h]
    elif w < h:
        # 计算垂直偏移量以使图像方正
        offset = (h - w) // 2

        # 裁剪高度以匹配宽度
        image = image[offset : offset + w]

    # 将图像调整为512x512像素
    image = np.array(Image.fromarray(image).resize((512, 512)))

    # 返回调整后的图像
    return image


def change_images_to_file(
    generated_images_path,
    image_name,
    num_steps,
):
    """
    将生成的图像按步骤切割并保存为单独的文件。

    参数:
        generated_images_path (str): 生成图像的目录路径。
        image_name (str): 生成图像的文件名。
        num_steps (int): 切割的步骤数。

    返回:
        None
    """
    # 打开生成的图像文件
    images = Image.open(os.path.join(generated_images_path, image_name))

    # 将图像转换为NumPy数组
    images = np.array(images)

    # 遍历每一步，显示进度条
    for i in trange(1, num_steps + 1):
        # 生成文件名，使用四位数填充
        fig_name = str(i).zfill(4) + ".png"

        # 生成切割后图像的保存路径
        splice_image_path = os.path.join(generated_images_path, fig_name)

        # 计算切割图像的左边界
        left = 522 * (i - 1)

        # 计算切割图像的右边界
        right = left + 512

        # 切割图像
        image = images[:, left:right, :]

        # 将切割后的图像转换为PIL Image
        image = Image.fromarray(image)

        # 保存切割后的图像
        image.save(splice_image_path)
