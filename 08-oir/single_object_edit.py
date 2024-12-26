# 导入操作系统接口模块，用于与操作系统进行交互
import os

# 设置环境变量 "HF_ENDPOINT" 为指定的URL
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置环境变量 "TOKENIZERS_PARALLELISM" 为 "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入PyTorch库，用于深度学习
import torch

# 导入操作系统模块（重复导入，建议移除多余的导入）
import os

# 导入YAML解析库，用于读取配置文件
import yaml

# 导入时间模块，用于时间相关操作
import time

# 导入NumPy库，用于数组和矩阵运算

# 从PIL库导入Image模块，用于图像处理
from PIL import Image

# 从PIL库导入ImageDraw和ImageFont模块，用于图像处理和绘制
from PIL import Image

# 从自定义工具模块导入生成候选图像的函数
from utils.candidate_images_generation import candidate_images_generation

# 从自定义工具模块导入选择最优候选图像的函数
from utils.optimal_candidate_selection import optimal_candidate_selection

# 从自定义采样器模块导入DDIM反演类
from sampler.ddim_inversion import DDIMInversion

# 从配置模块导入相关函数和变量
from configs.basic_config import (
    load_pipe,  # 稳定扩散管道对象
    NUM_DDIM_STEPS,  # DDIM反演的步数
    GUIDANCE_SCALE,  # 引导尺度，用于控制生成图像的指导强度
    clip_text_path,  # CLIP模型的路径
)


def main(args):
    """
    主函数，执行图像编辑的整个流程，包括反演、生成候选图像和选择最优图像。

    参数:
        args (dict): 从配置文件加载的参数字典，包含图像路径、提示词等信息。
    """

    # 记录开始时间
    start_time = time.time()
    time_cost = {}
    # 0. 用户输入的基本信息
    # 原始图像的路径
    image_path = args["image_path"]

    # 生成图像的保存路径
    generation_image_path = args["generation_image_path"]

    # 原始提示词
    origin_prompt = args["origin_prompt"]

    # 目标提示词列表
    target_prompt_list = args["target_prompt_list"]

    # 提示词变化列表
    prompt_changes = args["target_changes"]

    # 原始提示词的mask路径
    origin_prompt_mask = args["origin_prompt_mask"]

    # 1. 目标提示词的准备
    # 初始化目标提示词列表
    target_prompts_list = []

    for prompt_change in prompt_changes:
        # 将原始提示词、提示变化和目标提示词拼接，形成完整的目标提示词
        new_target_prompt = (
            target_prompt_list[0] + prompt_change + target_prompt_list[1]
        )
        target_prompts_list.append(new_target_prompt)

    # 2. 反演过程

    # 打印反演开始的信息
    print("Inversion ...")

    # 加载稳定扩散管道
    pipe = load_pipe()

    # 初始化DDIM反演对象，传入稳定扩散管道
    ddim_inversion = DDIMInversion(pipe)

    # 对原始图像进行反演，获取所有潜在表示
    all_latents = ddim_inversion.invert(image_path, origin_prompt)

    # 记录反演结束时间并计算耗时
    end_time = time.time()
    time_cost["inversion"] = end_time - start_time
    start_time = end_time

    # 3. 收集所有候选图像
    # 打印候选图像生成开始的信息
    print("Candidate images generation ...")

    # 初始化候选图像的字典
    candidate_images = {}

    for target_prompt, prompt_change in zip(target_prompts_list, prompt_changes):
        # 对每个目标提示词和对应的提示变化，生成候选图像
        candidate_images[prompt_change] = candidate_images_generation(
            pipe,  # 稳定扩散管道对象
            origin_prompt,  # 原始提示词
            target_prompt,  # 目标提示词
            prompt_change,  # 提示变化
            num_inference_steps=NUM_DDIM_STEPS,  # 推理步数
            guidance_scale=GUIDANCE_SCALE,  # 引导尺度
            all_latents=all_latents,  # 所有潜在表示
        )

    # 保存候选图像
    for prompt_change, images in candidate_images.items():
        # 创建保存目录
        os.makedirs(os.path.join(generation_image_path, prompt_change), exist_ok=True)

        for i, image in enumerate(images):
            # 保存每张候选图像为PNG文件
            Image.fromarray(image).save(
                os.path.join(generation_image_path, prompt_change, f"{i}.png")
            )

    # 删除模型引用并清理CUDA缓存
    del pipe
    torch.cuda.empty_cache()

    # 记录候选图像生成结束时间并计算耗时
    end_time = time.time()
    time_cost["candidate_images_generation"] = end_time - start_time
    start_time = end_time

    # 4. 选择最优的候选图像
    # 打印最优候选图像选择开始的信息
    print("Optimal candidate selection ...")

    # 初始化最优反演步数和所有mask的字典
    optimal_inversion_steps = {}
    all_masks = {}

    # 初始化非编辑区域的mask为1（全选）
    all_masks["non_editing_region_mask"] = 1

    for p_idx, prompt_change in zip(range(len(prompt_changes)), prompt_changes):
        # 遍历每个提示变化及其索引
        max_idx, output_image = optimal_candidate_selection(
            origin_image_path=image_path,  # 原始图像的路径
            editing_region_mask_path=origin_prompt_mask,  # 编辑区域mask的路径
            candidate_images=candidate_images[
                prompt_change
            ],  # 当前提示变化对应的候选图像列表
            target_prompt_change=prompt_change,  # 当前提示变化
            all_masks=all_masks,  # 所有mask的字典
            clip_text_path=clip_text_path,  # CLIP模型的路径
        )

        # 打印当前提示变化及其最优反演步数
        print(prompt_change, max_idx)

        # 将最优反演步数存储到字典中
        optimal_inversion_steps[prompt_changes[p_idx]] = max_idx

        # 如果保存路径不存在，创建目录
        if not os.path.exists(generation_image_path):
            os.makedirs(generation_image_path)

        # 将输出图像转换为PIL Image并保存到指定路径
        img = Image.fromarray(output_image).save(
            os.path.join(generation_image_path, f"{prompt_change}_{max_idx}_best.png")
        )

    # 记录最优候选选择结束时间并计算耗时
    end_time = time.time()
    time_cost["optimal_candidate_selection"] = end_time - start_time
    for key, value in time_cost.items():
        print(f"{key}: {int(value)} seconds")


if __name__ == "__main__":
    # 从YAML配置文件中加载参数
    with open("configs/single_object_edit.yaml", "r") as file:
        args = yaml.safe_load(file)

    # 对每个参数组执行主函数
    for key in args.keys():
        main(args[key])
