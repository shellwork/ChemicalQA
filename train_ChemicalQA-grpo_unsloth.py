
# 如果在使用过程中出现错误，请确保安装指定的 TRL 版本和 unsloth 库
# 可以使用以下命令进行安装：
# pip install trl==0.15.0
# pip install unsloth

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # 对 TRL 进行补丁处理

import logging
import os
import random
import re
import math
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List
from openai import OpenAI

from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

@dataclass
class DatasetArguments:
    """数据集参数的数据类"""

    # 数据集 ID 或路径
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    # 数据集拆分
    dataset_splits: str = "train"
    # 分词器名称或路径
    tokenizer_name_or_path: str = None

@dataclass
class SwanlabArguments:
    """SwanLab参数的数据类"""

    # 是否使用 SwanLab
    swanlab: bool
    # SwanLab 用户名
    workspace: str
    # SwanLab 的项目名
    project: str
    # SwanLab 的实验名
    experiment_name: str

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)  # 设置日志格式

logger.addHandler(handler)

def format_reward_func(completions, **kwargs):
    """
    格式奖励函数，检查模型输出格式是否匹配: <think>...</think><answer>...</answer>

    参数:
        completions (list[str]): 生成的输出
    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出
    for completion in completions:
        try:
            # 在生成的输出前添加<think>标签，便于后续正则表达式匹配
            completion = "<think>" + completion

            if random.random() < 0.1:  # 1% 的概率将生成输出写入文件
                # 创建生成输出目录（如果不存在）
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)  # 写入生成的输出

            # 定义正则表达式模式，用于匹配 <think> 和 <answer> 标签
            regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
            match = re.search(regex, completion, re.DOTALL) # 使用正则表达式进行匹配

            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)  # 如果格式不正确，奖励为 0
            else:
                rewards.append(1.0)  # 如果格式正确，奖励为 1
        except Exception:
            rewards.append(0.0)  # 如果发生异常，奖励为 0

    return rewards

client = OpenAI(
    api_key=os.getenv("API_KEY"),  # 可在此处替换API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def get_text_embedding(
    text: str,
    max_retries: int = 3,   # 最大重试次数
    base_delay: float = 1.0 # 首次重试的等待秒数
):
    """
    将文本转换为通用文本向量。
    使用通义实验室的 text-embedding-v3 模型，得到 1024 维 float 向量。
    如果所有重试都失败，则返回 1024 维零向量以避免训练中断。
    """
    if not text or not text.strip():
        return np.zeros((1024,), dtype=float)

    # 指数回退的延迟
    delay = base_delay

    for attempt in range(max_retries):
        try:
            # 调用通用文本向量 API
            response = client.embeddings.create(
                model="text-embedding-v3",  # 模型名称
                input=text,                 # 输入文本
                dimensions=1024,           # 指定返回向量维度，可选 1024/768/512
                encoding_format="float"    # 返回的向量编码格式
            )

            embedding = response.data[0].embedding
            return np.array(embedding, dtype=float)

        except Exception as e:
            logger.warning(
                f"调用 embeddings API 发生异常 (第 {attempt+1} 次)，错误信息: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                logger.error(f"已达到最大重试次数 ({max_retries})，将返回零向量以继续训练。")
                return np.zeros((1024,), dtype=float)


def cosine_similarity(vec_a, vec_b):
    """
    计算余弦相似度
    """
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def split_reasoning_by_steps(reasoning_text):
    """
    根据“步骤1/2/3/4” 将推理文本拆分为 4 部分并返回一个 dict:
      {
         1: "...",
         2: "...",
         3: "...",
         4: "..."
      }
    如果匹配不到某个步骤，则其文本部分可置空。
    注意：这里的正则匹配/字符串处理逻辑可根据具体格式再做精调。
    """

    pattern = r"(\*\*步骤[1-4]：.*?\*\*)(.*?)((?=\*\*步骤[1-4]：)|$)"
    matches = re.findall(pattern, reasoning_text, flags=re.DOTALL | re.MULTILINE)

    # 存储结果
    step_contents = {1: "", 2: "", 3: "", 4: ""}
    for match in matches:
        step_header = match[0]  # "**步骤X：xxx**"
        step_body   = match[1]  # 对应的内容
        # 判断是哪一个步骤
        step_num_search = re.search(r"\*\*步骤([1-4])：", step_header)
        if step_num_search:
            step_num = int(step_num_search.group(1))
            # 去掉首尾空白
            step_contents[step_num] = step_header + step_body
    return step_contents

def check_format(think_text):
    """
    检查推理格式：
      - 是否有大步骤：步骤1,2,3,4
      - 每个步骤下是否有对应的小标题等
    满足则得相应分值(每个大步骤 0.25 分，否则 0 分)
    """
    score = 0.0
    # 按照要求的“步骤1...”、“步骤2...”、“步骤3...”、“步骤4...”等做正则/字符串查找
    # 如果有多个小标题要求，比如步骤1:
    #   "**步骤1：结构解析**"
    #   "1. 核心骨架识别："
    #   "2. 关键官能团标注："
    # 则只有全部匹配到才给 0.25 分，否则不给分
    
    step1_patterns = [
        r"\*\*步骤1：结构解析\*\*",
        r"1\. 核心骨架识别：",
        r"2\. 关键官能团标注："
    ]
    step2_patterns = [
        r"\*\*步骤2：物化性质与功能关联\*\*",
        r"1\. 性质推断：",
        r"2\. 关联问题关键词："
    ]
    step3_patterns = [
        r"\*\*步骤3：选项分析与排除\*\*",
        r"1\. 正确选项：",
        r"2\. 错误选项："
    ]
    step4_patterns = [
        r"\*\*步骤4：结论\*\*"
    ]
    
    # 定义一个辅助函数检查一整组模式
    def check_step_patterns(text, patterns):
        for p in patterns:
            if not re.search(p, text):
                return False
        return True

    # 逐步检查
    if check_step_patterns(think_text, step1_patterns):
        score += 0.25
    if check_step_patterns(think_text, step2_patterns):
        score += 0.25
    if check_step_patterns(think_text, step3_patterns):
        score += 0.25
    if check_step_patterns(think_text, step4_patterns):
        score += 0.25

    return score

def chemical_reward_func(prompts, completions, content, reasoning_content, **kwargs):
    """
    基于新需求的奖励函数示例:
      prompts: 问题列表
      completions: 模型给出的回答列表
      content: 正确答案(如 "C")
      reasoning_content: 数据集中标准推理文本

    返回:
      [float, float, ...]  # 与输入数量相同的奖励值列表
    """
    rewards = []
    for prompt, completion, content, reasoning_content in zip(prompts, completions, content, reasoning_content):
        # 1) 解析出 <think>...</think> 和 <answer>...</answer> 的内容
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

        if not think_match:
            # 如果没有 <think>，则格式分肯定是 0，推理相似度也无法做
            # 答案分若有 <answer> 则单独判断
            format_score = 0.0
            reasoning_score = 0.0
        else:
            # （a）推理格式检查
            think_text = think_match.group(1)
            format_score = check_format(think_text)  # 0~1 之间

            # （b）推理内容相似度
            #     先拆分模型输出的 <think> 内容，再拆分参考 reason_content
            model_steps = split_reasoning_by_steps(think_text)
            ref_steps   = split_reasoning_by_steps(reasoning_content)
            
            # 对 4 个步骤分别计算相似度
            step_scores = []
            for step_id in [1, 2, 3, 4]:
                model_emb = get_text_embedding(model_steps[step_id])
                ref_emb   = get_text_embedding(ref_steps[step_id])
                sim = cosine_similarity(model_emb, ref_emb)
                # 如果该步骤相似度 >= 0.8，则记 0.25，否则 0
                step_scores.append(0.25 if sim >= 0.8 else 0.0)
            reasoning_score = sum(step_scores)  # 0~1

        if not answer_match:
            # 如果没有 <answer> 则无法匹配答案，答案分=0
            answer_score = 0.0
        else:
            pred_answer = answer_match.group(1).strip()
            # 与 data["content"] 比较
            correct_answer = content.strip()
            if pred_answer == correct_answer:
                answer_score = 1.0
            else:
                answer_score = 0.0

        # 三项得分都在 0~1 区间，按权重 0.5/0.25/0.25 合并
        #   reasoning_score * 0.5 + answer_score * 0.25 + format_score * 0.25
        final_score = reasoning_score * 0.5 + answer_score * 0.25 + format_score * 0.25
        rewards.append(final_score)
        
        if final_score >= 0.8:
            if random.random() < 0.10:
                    # 创建生成输出目录（如果不存在）
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"Prompt: {prompt}\nCompletion: {completion}\nFormat_score={format_score}, Reasoning_score={reasoning_score}, Answer_score={answer_score},Final_score={final_score}\n")

        # 打印调试信息
        if random.random() < 0.3:
            print(f"Prompt: {prompt}\nCompletion: {completion}")
            print(f"Format_score={format_score}, Reasoning_score={reasoning_score}, Answer_score={answer_score}")
            print(f"Final_score={final_score}\n")

    return rewards


def get_checkpoint(training_args: GRPOConfig):
    """
    获取最后一个检查点

    参数:
        training_args (GRPOConfig): 训练参数
    返回:
        str: 最后一个检查点的路径，如果没有检查点，则返回 None
    """
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):  # 如果输出目录存在
        # 获取最后一个检查点
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

# 定义 GRPO 训练函数
def grpo_function(
    model_args: ModelConfig,
    dataset_args: DatasetArguments,
    training_args: GRPOConfig,
    callbacks: List,
):
    # 记录模型参数
    logger.info(f"Model parameters {model_args}")
    # 记录训练/评估参数
    logger.info(f"Training/evaluation parameters {training_args}")

    # 从预训练模型加载模型和分词器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,  # 模型名称或路径
        fast_inference=True,  # 启用 vLLM 快速推理
        load_in_4bit=True,  # 是否以 4 位加载模型，False 表示使用 LoRA 16 位
        max_lora_rank=model_args.lora_r,  # 设置 LoRA 的最大秩
        max_seq_length=training_args.max_completion_length,  # 设置最大序列长度
        gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,  # GPU 内存利用率
        attn_implementation=model_args.attn_implementation, # 设置注意力实现方式 flash attention
    ) 

    # PEFT 模型
    model = FastLanguageModel.get_peft_model(
        model,
        r = model_args.lora_r,  # 选择任意大于 0 的数字！建议使用 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # 如果内存不足，可以移除 QKVO
        lora_alpha = model_args.lora_alpha,  # 设置 LoRA 的 alpha 值
        use_gradient_checkpointing = "unsloth",  # 启用长上下文微调
        random_state = training_args.seed,  # 设置随机种子
    )

    # 如果分词器没有填充标记，则使用结束标记作为填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    dataset = load_dataset("json", data_files=dataset_args.dataset_id_or_path, split="train")
    dataset = dataset.remove_columns(["cid"])  # 确保 dataset 里不含 cid

    # 随机选择 10K 个样本
    # dataset = dataset.shuffle(seed=training_args.seed).select(range(10000))

    def generate_r1_prompt(data):
        """
        生成 R1 Chemical QA提示词

        参数:
            smiles (string): 代表化学结构的字符串
            content (char): 选择题答案
        返回:
            dict: 生成的一个数据样本
        """
        # 定义提示词前缀
        r1_prefix = [
            {
                "role": "user",
                "content": f"""用给定的分子表示法 (SMILES) {data['smiles']}，完成以下化学问题：
问题 (Question)：{data['input']}
正确答案 (Answer)：{data['content']}

你是一名化学专家，请按照以下 4 个步骤展开思考，并在 <think> ... </think> 标签中完整展示你的推理过程；最后请在 <answer> ... </answer> 标签中给出最终答案,如 <answer>C</answer>。在 <think> 部分，务必进行逐步思考、结构分析以及对错误选项的排除，要求如下：

步骤1：结构解析
1. 核心骨架识别：
指出环系或骨架(如苯环、杂环、β-内酰胺等)
标明立体化学（@/@@等）或顺反异构
2. 关键官能团标注：
列出功能基团（如酯基-COO、羟基-OH 等）
强调影响性质或活性的基团（如易水解基团、极性基团）
步骤2：物化性质与功能关联
1. 性质推断：
极性或亲脂性(logP 趋势)
溶解度(离子化基团 vs. 非极性基团)
2. 关联问题关键词：
若问题涉及吸收/代谢/活性，需重点分析相关基团
步骤3：选项分析与排除
1. 正确选项：
说明关键结构特征与正确答案的对应关系
2. 错误选项：
逐条反驳（为什么与该分子不符）
步骤4：结论
简要总结，说明为什么只剩下该正确答案
在 <think> 标签中：
合并上述 4 步分析为完整推理过程
用 **加粗**来标记关键术语或官能团(如 酯键、logP)
适度精简，不超过 3~4 行要点即可
在 <answer> 标签中：
仅写最终答案（例如 <answer>C</answer>），无需重复分析

示例：
<think>
**步骤1：结构解析**
1. **核心骨架识别**：
- 具体分析内容省略
2. **关键官能团标注**： 
- 具体分析内容省略

**步骤2：物化性质与功能关联**
1. **母核推断**：
- 具体分析内容省略

**步骤3：选项分析与排除**
1. **正确选项C**：
- 具体分析内容省略
2. **错误选项排除**：
- **A（p-menthane）**：具体分析内容省略
- **B（daidzein）**：具体分析内容省略
- **D（hesperetin）**：具体分析内容省略

**步骤4：结论**
- 具体分析内容省略
</think>
<answer>正确答案</answer>

现在请根据以上要求，在 <think> 中展示详细推理，在 <answer> 中填写最终答案。""",
            },
            {
                "role": "assistant",
                "content": "让我们逐步解决这个问题。\n<think>",  # 结尾使用 `<think>` 促使模型开始思考
            },
        ]
        
        # 先把 r1_prefix 转成最终字符串
        prompt_str = tokenizer.apply_chat_template(
            r1_prefix, 
            tokenize=False, 
            continue_final_message=True
        ) + tokenizer.eos_token

        # 接下来将字符串进行分词，检查长度
        tokens = tokenizer(prompt_str, add_special_tokens=False).input_ids
        max_len = training_args.max_completion_length  # 这里默认用 training_args 的 max长度
        if len(tokens) > max_len:
            # 截断到 max_len
            tokens = tokens[:max_len]
            # 记录日志
            logger.warning(
                f"Prompt token length {len(tokens)} exceeded {max_len}, truncating..."
            )

        # 再 decode 回字符串
        truncated_prompt_str = tokenizer.decode(tokens, skip_special_tokens=False)
        
        return {
        "prompt": truncated_prompt_str,
        "content": data["content"],
        "reasoning_content": data.get("reasoning_content", ""),
        # "cid": data["cid"],
    }

    # 将数据集转换为 R1 Chemical QA 提示词
    dataset = dataset.map(generate_r1_prompt)
    # 将数据集拆分为训练集和测试集，拆分比例为 9:1
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]  # 训练集
    test_dataset = train_test_split["test"]    # 测试集
    
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
   

    # 设置 GRPOTrainer
    trainer = GRPOTrainer(
        model = model,
        # model=model_args.model_name_or_path,  # 模型名称或路径
        # 奖励函数列表，用于计算奖励分数
        reward_funcs=[
            format_reward_func,  # 格式奖励函数
            chemical_reward_func,  # QA奖励函数
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )


    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )

    # 训练模型
    train_result = trainer.train()

    # 记录和保存指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # 保存模型和分词器
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    model.save_lora(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")
    logger.info("*** Training complete! ***")

def main():
    """主函数，用于执行主训练循环"""
    # 解析命令行参数和配置文件
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig, SwanlabArguments))
    model_args, dataset_args, training_args, swanlab_args = (
        parser.parse_args_and_config()
    )

    # 如果使用 SwanLab，则创建 SwanLab 回调对象，用于训练信息记录
    if swanlab_args.swanlab:
        swanlab_callback = SwanLabCallback(
            workspace=swanlab_args.workspace,
            project=swanlab_args.project,
            experiment_name=swanlab_args.experiment_name,
        )
        callbacks = [swanlab_callback]
    else:
        callbacks = None

    # 运行主训练循环
    grpo_function(model_args, dataset_args, training_args, callbacks=callbacks)

if __name__ == "__main__":
    main()
