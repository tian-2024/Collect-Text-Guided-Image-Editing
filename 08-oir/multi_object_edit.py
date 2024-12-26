# 导入操作系统接口模块，用于与操作系统进行交互
import os

# 设置环境变量 "HF_ENDPOINT" 为指定的URL
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置环境变量 "TOKENIZERS_PARALLELISM" 为 "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入必要的库和模块，包括系统模块、PyTorch、OpenCV、YAML解析、文件操作、命令行解析和时间模块
import yaml
import time

# 导入NumPy库，用于数组和矩阵运算

# 从PIL库导入Image模块，用于图像处理
from PIL import Image

# 从tqdm库导入用于Jupyter Notebook的进度条

# 导入Matplotlib的pyplot模块，用于绘图

# 从配置模块导入相关函数和变量
from configs.basic_config import (
    load_pipe,
    NUM_DDIM_STEPS,  # DDIM反演的步数
    GUIDANCE_SCALE,  # 引导尺度，用于控制生成图像的指导强度
    clip_text_path,  # CLIP模型的路径
)

# 从自定义工具模块导入candidate_images_generation函数，用于生成候选图像
from utils.candidate_images_generation import candidate_images_generation

# 从自定义工具模块导入optimal_candidate_selection函数，用于选择最优候选图像
from utils.optimal_candidate_selection import optimal_candidate_selection

# 从自定义工具模块导入oir函数，用于执行Optimal Inversion Refinement过程
from utils.oir import oir

# 从自定义采样器模块导入DDIMInversion类，用于进行DDIM反演
from sampler.ddim_inversion import DDIMInversion


def main(args):
    """
    主函数，执行图像编辑的整个流程，包括反演、生成候选图像和选择最优图像。

    参数:
        args (dict): 从配置文件加载的参数字典，包含图像路径、提示词等信息。
    """
    # 初始化时间记录字典
    time_cost = {}

    # 记录开始时间
    start_time = time.time()

    # 0. 用户输入的基本信息
    # 原始图像的路径
    image_path = args["image_path"]

    # 原始提示词
    origin_prompt = args["origin_prompt"]

    # 目标提示词
    target_prompt = args["target_prompt"]

    # 引导提示词列表
    guided_prompts = args["guided_prompt"]

    # 原始变化（未在后续代码中使用）
    origin_changes = args["origin_change"]

    # 提示词变化列表
    prompt_changes = args["prompt_change"]

    # 提示词变化对应的mask路径列表
    prompt_changes_mask = args["prompt_change_mask"]

    # 重新组装的步数
    reassembly_step = args["reassembly_step"]

    # 反演步数
    reinversion_steps = args["reinversion_steps"]

    # 生成图像的保存路径
    generation_image_path = args["generation_image_path"]

    # 1. 引导提示词的准备
    # 初始化引导提示词列表和总提示词列表（包含原始提示词）
    guided_prompts_list = []
    prompts = [origin_prompt]

    for guided_prompt, prompt_change in zip(guided_prompts, prompt_changes):
        # 将引导提示词与对应的变化进行拼接，形成新的引导提示词，并添加到引导提示词列表中
        new_guided_prompt = guided_prompt[0] + prompt_change + guided_prompt[1]
        guided_prompts_list.append(new_guided_prompt)

    for prompt in guided_prompts_list:
        # 将所有引导提示词添加到总提示词列表中
        prompts.append(prompt)

    # 将目标提示词添加到总提示词列表的末尾
    prompts.append(target_prompt)

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
    time_cost["Inversion"] = end_time - start_time
    start_time = end_time  # 更新开始时间

    # 3. 收集所有候选图像，并保存到文件
    # 打印候选图像生成开始的信息
    print("Candidate images generation ...")

    # 初始化候选图像的字典
    candidate_images = {}

    for guided_prompt, prompt_change in zip(guided_prompts_list, prompt_changes):
        # 对每个引导提示词和对应的提示变化，生成候选图像并存储在字典中
        candidate_images[prompt_change] = candidate_images_generation(
            pipe,  # 稳定扩散管道对象
            origin_prompt,  # 原始提示词
            guided_prompt,  # 当前引导提示词
            prompt_change,  # 当前提示变化
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

    # 记录候选图像生成结束时间并计算耗时
    end_time = time.time()
    time_cost["Candidate-images-generation"] = end_time - start_time
    start_time = end_time  # 更新开始时间

    # 4. 选择候选图像中的最优反演步数
    # 打印最优候选选择开始的信息
    print("Optimal candidate selection ...")

    # 初始化最优反演步数字典和所有mask的字典
    optimal_inversion_steps = {}
    all_masks = {}

    # 设置非编辑区域mask为1（可能需要根据实际情况调整）
    all_masks["non_editing_region_mask"] = 1

    for p_idx, (prompt_change, prompt_change_mask) in enumerate(
        zip(prompt_changes, prompt_changes_mask)
    ):
        # 遍历每个提示变化及其对应的mask路径
        max_idx, _ = optimal_candidate_selection(
            origin_image_path=image_path,  # 原始图像路径
            editing_region_mask_path=prompt_change_mask,  # 当前提示变化的mask路径
            candidate_images=candidate_images[
                prompt_change
            ],  # 当前提示变化对应的候选图像列表
            target_prompt_change=prompt_change,  # 当前提示变化
            all_masks=all_masks,  # 所有mask的字典
            clip_text_path=clip_text_path,  # CLIP模型路径
        )
        # 将选择的最优反演步数存储到字典中
        optimal_inversion_steps[prompt_change] = max_idx

    # 5. 确保最优反演步数按从小到大排列，并获取所有mask
    # 按最优反演步数对提示变化进行排序
    prompt_changes = sorted(prompt_changes, key=lambda x: optimal_inversion_steps[x])

    # 设置最大最优反演步数的mask和初始化所有编辑区域mask
    all_masks["max_optimal_inversion_step_mask"] = all_masks[prompt_changes[-1]]
    all_masks["all_editing_region_mask"] = 0

    for prompt_change in prompt_changes:
        # 累加所有编辑区域mask
        all_masks["all_editing_region_mask"] += all_masks[prompt_change]

    # 记录最优候选选择结束时间并计算耗时
    end_time = time.time()
    time_cost["Optimal-candidate-selection"] = end_time - start_time
    start_time = end_time  # 更新开始时间

    # 6. 实现Optimal Inversion Refinement (OIR)
    # 打印OIR开始的信息
    print("OIR ...")

    # 获取最大的最优反演步数
    max_optimal_inversion_step = optimal_inversion_steps[prompt_changes[-1]]

    # 获取对应步数的潜在表示
    x_t = all_latents[max_optimal_inversion_step]

    # 执行OIR过程
    images, x_t = oir(
        pipe,  # 稳定扩散管道对象
        prompts,  # 总提示词列表
        optimal_inversion_steps=optimal_inversion_steps,  # 最优反演步数字典
        x_t=x_t,  # 当前潜在表示
        num_inference_steps=NUM_DDIM_STEPS,  # 推理步数
        guidance_scale=GUIDANCE_SCALE,  # 引导尺度
        all_latents=all_latents,  # 所有潜在表示
        all_masks=all_masks,  # 所有mask
        ddim_inversion=ddim_inversion,  # DDIM反演对象
        reinversion_steps=reinversion_steps,  # 反演步数
        reassembly_step=reassembly_step,  # 重新组装步数
        prompt_changes=prompt_changes,  # 提示变化列表
    )

    # 保存生成的图像
    if not os.path.exists(generation_image_path):
        # 如果保存路径不存在，创建目录
        os.makedirs(generation_image_path)

    # 将生成的图像保存为PNG文件
    Image.fromarray(images.squeeze(0)).save(
        os.path.join(generation_image_path, target_prompt + ".png")
    )

    # 记录OIR结束时间并计算耗时
    end_time = time.time()
    time_cost["OIR"] = end_time - start_time

    # 打印各步骤的耗时
    for key, value in time_cost.items():
        print(f"{key}: {int(value)} seconds")


if __name__ == "__main__":
    # 从YAML配置文件中加载参数
    with open("configs/multi_object_edit.yaml", "r") as file:
        args = yaml.safe_load(file)

    # 对每个配置项调用主函数
    for key in args.keys():
        main(args[key])
