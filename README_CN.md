# GRPO强化学习项目：结构推理与化学问题求解

本项目基于TRL库的GRPO（Generalized Reinforcement Policy Optimization）方法，使用Unsloth对TRL进行了优化，以实现快速强化学习训练。由于Unsloth框架能够大幅降低显存占用，作者能够在单张4090显卡（本项目使用了A10 24G显卡）上成功实现GRPO强化学习训练。项目目标是训练一个强化学习（RL）微调后的语言模型，通过特定的推理格式指导和奖励函数，提高模型在化学问题（如分子结构识别、官能团标注、物化性质关联、选项分析）的推理准确性。

## 项目主要内容

- **模型与库使用**：使用`unsloth`库对TRL的GRPO算法进行高效实现，优化训练过程。
- **数据集**：数据集基于MoleculeQA，通过调用Deepseek-R1 API，根据特定提示词获取了Deepseek-R1对于该数据集中化学问题的详细推理过程，经过处理整理成了`train.jsonl`数据集。
- **推理指导与格式验证**：模型的输出需符合特定的推理格式（`<think>`和`<answer>`标签），通过正则表达式严格验证格式并给予奖励。
- **嵌入与相似度评分**：使用阿里云通义千问文本嵌入模型 (`text-embedding-v3`)，对模型输出与标准答案进行语义相似度评估。
- **GRPO训练流程**：利用`unsloth`库提高训练效率，应用LoRA（低秩自适应）和 GRPO（Generalized Reward-Policy Optimization）技术高效地进行强化学习微调。

## 奖励函数说明

项目中设计了两种奖励函数，以优化模型的输出质量：

- **格式奖励 (`format_reward_func`)**：
  - 验证输出是否严格符合特定格式（即必须包含`<think>`和`<answer>`标签）。
  - 格式正确得1分，格式错误得0分。

- **答案奖励 (`chemical_reward_func`)**：
  - 将模型输出与标准答案进行对比。
  - 考察内容包括推理结构（需包含步骤1-4的结构化推理过程）与推理内容相似度（通过文本嵌入与余弦相似度评估），以及答案准确性。
  - 各部分权重为：推理内容相似度60%，答案准确性30%，格式正确性10%。

## 项目成果

- 实现了基于`unsloth`和TRL的强化学习微调流程。
- 设计了一套完备的推理过程验证与奖励机制，提高了模型对化学领域问题的精确解答能力。
- 提供了详细日志输出，支持模型训练过程的监控和结果分析。
- 最终模型能够以高准确性完成标准化的化学问题推理，确保输出质量和格式一致性。

## 快速开始

克隆仓库代码并安装依赖：

```bash
git clone https://github.com/shellwork/ChemicalQA
cd ChemicalQA
bash install.sh
```

- `install.sh` 将安装项目额外依赖（适合于常用的模型训练云环境，预装有适合版本的torch、cuda和cudnn）。

编辑 `train_ChemicalQA-grpo_unsloth.sh` 文件，填入你自己的文本嵌入模型API（如阿里云千问API），然后执行以下命令开始训练：

```bash
bash train_ChemicalQA-grpo_unsloth.sh
```

训练参数和训练曲线可在 SwanLab 查看：

- [SwanLab项目链接](https://swanlab.cn/@shellwork/ChemicalQA-gpro/overview)

## 项目代码与数据

- 所有相关代码已开源在GitHub：[ChemicalQA](https://github.com/shellwork/ChemicalQA)

## 日志与输出
- 模型训练过程的日志会实时输出到控制台，并保存部分推理示例至`completion_samples`目录下，以便于结果分析。

更多细节请参阅代码文件内注释。

