# 导入PyTorch库
import torch

# 导入NumPy库，用于数组和矩阵运算
import numpy as np

# 从PIL库导入Image模块，用于图像处理
from PIL import Image

# 导入PyTorch的神经网络功能模块
import torch.nn.functional as F

# 从自定义模块导入MaskCLIPModel类
from models.mask_clip import MaskCLIPModel

# 从transformers库导入AutoProcessor、CLIPModel和AutoTokenizer，用于处理和使用预训练的CLIP模型
from transformers import AutoProcessor, AutoTokenizer


def change_mask_to_attention_mask(
    mask, patch_size=16, image_size=224, use_cls_token=True
):
    """
    将二值化的mask转换为适用于注意力机制的attention mask。

    参数:
        mask (torch.Tensor): 输入的二值化mask，形状为(H, W)。
        patch_size (int): 每个patch的大小，默认为16。
        image_size (int): 输入图像的大小，默认为224。
        use_cls_token (bool): 是否使用CLS token，默认为True。

    返回:
        attention_mask (torch.Tensor): 生成的注意力mask，形状为(1, 1, seq_len, seq_len)。
    """
    # 计算图像被划分为多少个patch的高度和宽度
    h, w = image_size // patch_size, image_size // patch_size

    # 计算序列长度，是否包含CLS token
    seq_len = h * w + 1 if use_cls_token else h * w

    # 增加两个维度，使mask形状变为(N=1, C=1, H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)

    # 将mask转换为浮点类型，并调整大小为(h, w)，使用最近邻插值方式
    mask = (
        torch.nn.functional.interpolate(
            mask.float(),
            size=(h, w),
            mode="nearest",
        )
        > 0.5
    )

    # 去掉通道维度，mask形状变为(N=1, H, W)
    mask = mask.squeeze(1)

    # 将mask展平成(N, H*W)
    mask = torch.flatten(mask, start_dim=1)

    # 创建CLS token的mask，形状为(N, 1)
    cls_token = torch.ones_like(mask[:, :1])

    # 将CLS token的mask与展平后的mask连接，形状为(N, seq_len)
    mask = torch.concat((cls_token, mask), dim=1)

    # 初始化注意力mask，形状为(1, 1, seq_len, seq_len)
    attention_mask = torch.zeros((1, 1, seq_len, seq_len))

    # 遍历每个位置并设置注意力mask
    for i in range(seq_len):
        for j in range(seq_len):
            if mask[:, i] and mask[:, j]:
                attention_mask[:, :, i, j] = 1

    # 返回生成的注意力mask
    return attention_mask


def change_mask_shape(mask, image):
    """
    调整mask的形状以匹配图像的尺寸，并扩展到3个通道。

    参数:
        mask (torch.Tensor): 输入的二值化mask，形状为(H, W)。
        image (torch.Tensor): 输入的图像，形状为(N, C, H, W)。

    返回:
        new_mask (torch.Tensor): 调整后的mask，形状为(N, 3, H, W)。
    """
    # 获取图像的高度和宽度
    _, _, h, w = image.shape

    # 增加一个批次维度并重复3次以匹配图像的通道数，形状为(N, 3, H, W)
    mask = mask.unsqueeze(0).repeat(1, 3, 1, 1)

    # 将mask转换为浮点类型，并调整大小为图像的高度和宽度，使用最近邻插值方式
    new_mask = (
        torch.nn.functional.interpolate(
            mask.float(),
            size=(h, w),
            mode="nearest",
        )
        > 0.5
    )

    # 返回调整后的mask
    return new_mask


def editing_region_clip_score(
    target_prompt_change,
    candidate_images,
    editing_region_mask,
    clip_text_path,
):
    """
    计算编辑区域的CLIP评分，用于衡量候选图像与目标提示的匹配程度。

    参数:
        target_prompt_change (str): 目标提示的变化描述。
        candidate_images (list): 候选图像的列表。
        editing_region_mask (torch.Tensor): 编辑区域的mask，形状为(H, W)。
        clip_text_path (str): CLIP模型的路径。

    返回:
        scores (list): 每个候选图像的CLIP评分。
    """
    # 初始化分数列表
    scores = []

    # 加载预训练的MaskCLIP模型
    model = MaskCLIPModel.from_pretrained(clip_text_path)

    # 加载预训练的处理器
    processor = AutoProcessor.from_pretrained(clip_text_path)

    # 加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained(clip_text_path)

    # 对目标提示进行分词和编码
    target_prompt_change = tokenizer(
        target_prompt_change, padding=True, return_tensors="pt"
    )

    # 获取目标提示的文本特征
    target_prompt_change_feature = model.get_text_features(**target_prompt_change)

    # 检查CLIP模型的名称是否为clip-vit-base-patch16
    if clip_text_path.split("/")[-1] == "clip-vit-base-patch16":
        # 使用默认patch_size=16生成注意力mask
        editing_region_self_attention_mask = change_mask_to_attention_mask(
            editing_region_mask
        )
    elif clip_text_path.split("/")[-1] == "clip-vit-large-patch14":
        # 使用patch_size=14生成注意力mask
        editing_region_self_attention_mask = change_mask_to_attention_mask(
            editing_region_mask, patch_size=14
        )

    # 处理当前候选图像
    target_image = processor(images=candidate_images, return_tensors="pt")

    # 获取编辑区域的图像特征
    editing_region_of_target_image_feature = model.get_mask_image_features(
        **target_image,
        image_attention_mask=editing_region_self_attention_mask.repeat(
            len(candidate_images), 1, 1, 1
        )
    )

    # 计算文本特征与图像特征的余弦相似度
    scores = torch.cosine_similarity(
        target_prompt_change_feature.repeat(len(candidate_images), 1),
        editing_region_of_target_image_feature,
    )

    # 返回所有候选图像的评分
    return scores.tolist()


def non_editing_region_negative_MSE(
    origin_image,
    editing_region_mask,
    candidate_images,
    clip_text_path,
):
    """
    计算非编辑区域的负均方误差（MSE），用于衡量候选图像在非编辑区域与原图的相似度。

    参数:
        origin_image (numpy.ndarray): 原始图像。
        editing_region_mask (torch.Tensor): 编辑区域的mask，形状为(H, W)。
        candidate_images (list): 候选图像的列表。
        clip_text_path (str): CLIP模型的路径。

    返回:
        scores (list): 每个候选图像的负MSE评分。
    """
    # 初始化分数列表
    scores = []

    # 加载预训练的处理器
    processor = AutoProcessor.from_pretrained(clip_text_path)

    # 处理原始图像并获取像素值
    origin_image = processor(images=origin_image, return_tensors="pt")["pixel_values"]

    # 生成非编辑区域的mask
    non_editing_region_mask = change_mask_shape(1 - editing_region_mask, origin_image)

    # 获取原图的非编辑区域
    non_editing_region_of_origin_image = origin_image * non_editing_region_mask

    # 处理当前候选图像并获取像素值
    target_image = processor(images=candidate_images, return_tensors="pt")[
        "pixel_values"
    ]

    # 获取候选图像的非编辑区域
    non_editing_region_of_target_image = target_image * non_editing_region_mask

    # 计算负MSE作为评分
    scores = (
        F.mse_loss(
            non_editing_region_of_origin_image.repeat(len(candidate_images), 1, 1, 1),
            non_editing_region_of_target_image,
            reduction="none",
        ).mean(dim=(1, 2, 3))
        * -1
    )

    # 返回所有候选图像的评分
    return scores.tolist()


def optimal_candidate_selection(
    origin_image_path,
    target_prompt_change,
    editing_region_mask_path,
    candidate_images,
    all_masks,
    clip_text_path,
):
    """
    选择最优的候选图像，基于编辑区域的CLIP评分和非编辑区域的负MSE评分进行综合评估。

    参数:
        origin_image_path (str): 原始图像的路径。
        target_prompt_change (str): 目标提示的变化描述。
        editing_region_mask_path (str): 编辑区域mask的路径。
        candidate_images (list): 候选图像的列表。
        all_masks (dict): 所有编辑区域mask的字典。
        clip_text_path (str): CLIP模型的路径。

    返回:
        max_search_metric_idx (int): 最优候选图像的索引（从0开始）。
        candidate_images[max_search_metric_idx] (PIL.Image): 最优的候选图像。
    """
    # 打开并转换原始图像为numpy数组
    origin_image = np.array(Image.open(origin_image_path))

    # 打开并转换编辑区域mask为torch.Tensor，二值化处理
    editing_region_mask = torch.from_numpy(
        np.where(np.array(Image.open(editing_region_mask_path)) >= 1, 1, 0)
    )

    # 默认设置：不同编辑对的mask没有重叠
    # 将当前编辑区域mask添加到all_masks字典中，形状为(1, H, W)
    all_masks[target_prompt_change] = editing_region_mask.unsqueeze(0)

    # 更新非编辑区域mask，去除当前编辑区域
    all_masks["non_editing_region_mask"] -= editing_region_mask.unsqueeze(0)

    # 计算所有候选图像的非编辑区域负MSE评分
    non_editing_region_scores = non_editing_region_negative_MSE(
        origin_image,
        editing_region_mask,
        candidate_images,
        clip_text_path,
    )

    # 计算所有候选图像的编辑区域CLIP评分
    editing_region_scores = editing_region_clip_score(
        target_prompt_change,
        candidate_images,
        editing_region_mask,
        clip_text_path,
    )

    # 获取非编辑区域评分的最大值和最小值
    max_non_editing_region_score = max(non_editing_region_scores)
    min_non_editing_region_score = min(non_editing_region_scores)

    # 获取编辑区域评分的最大值和最小值
    max_editing_region_score = max(editing_region_scores)
    min_editing_region_score = min(editing_region_scores)

    # 归一化非编辑区域评分
    normalize_non_editing_region_scores = [
        (score - min_non_editing_region_score)
        / (max_non_editing_region_score - min_non_editing_region_score)
        for score in non_editing_region_scores
    ]

    # 归一化编辑区域评分
    normalize_editing_region_scores = [
        (score - min_editing_region_score)
        / (max_editing_region_score - min_editing_region_score)
        for score in editing_region_scores
    ]

    # 综合评分，非编辑区域评分加上编辑区域评分
    search_metric = [
        normalize_non_editing_region_scores[i] + normalize_editing_region_scores[i]
        for i in range(len(normalize_editing_region_scores))
    ]

    # 找到综合评分最高的索引
    max_search_metric_idx = search_metric.index(max(search_metric))

    # 返回最优候选图像的索引（从0开始）和图像
    return (
        max_search_metric_idx,
        candidate_images[max_search_metric_idx],
    )
