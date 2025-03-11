import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# 1. 指定模型路径
# -----------------------------
base_model_path = "/mnt/workspace/model/Qwen2___5-3B-Instruct"  # 例如 "facebook/opt-1.3b"
lora_model_path = "/mnt/workspace/R1_zero/output/ChemicalQA-grpo"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. 加载分词器
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# 某些模型可能需要额外参数，如:  tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 3. 加载基础模型（原模型）
# -----------------------------
# 如果显存不足，可以尝试 device_map="auto"、torch_dtype=torch.float16 等
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,     # 使用半精度来减小显存占用
    device_map="auto",            # 自动将权重映射到可用的GPU上
    trust_remote_code=True        # 如果使用了自定义模型，需要打开此选项
)

# -----------------------------
# 4. 加载 LoRA 权重
# -----------------------------
model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    torch_dtype=torch.float16
)
# 如果你想直接“合并”LoRA权重到基础模型中，也可以使用：
# model = model.merge_and_unload()

# 若不需要 merge，则可直接在推理阶段使用 PEFT 形式进行推断。
# 下面为了演示，这里选择 merge 到原模型后再执行推理。
model = model.merge_and_unload()

model.to(device)
model.eval()

# -----------------------------
# 5. 构造推理输入
# -----------------------------
prompt = """用给定的分子表示法 (SMILES) C/C(=C\\CC/C(C)=C/CO)CO，完成以下化学问题：
问题 (Question)：Please complete the following question answering task: You are given a 5363397 of a molecule and a question about it with several options, please analysis the structure of the molecule and choose the right answer for the question from given options.\nMolecule 5363397: C/C(=C\\CC/C(C)=C/CO)CO\nQuestion about this molecule: What components does this molecule have?\nOption A: Geraniol bearing a hydroxy substituent at position 8.\nOption B: It consists of arachidonic acid bearing an additional hydroxy substituent at position 17.\nOption C: It consists of prostaglandin H1 bearing an additional hydroxy substituent at position 19.\nOption D: It consists of arachidonic acid bearing a hydroxy substituent at position 19.\n.

你是一名化学专家，请按照以下 4 个步骤展开思考，并在 <think> ... </think> 标签中完整展示你的推理过程；最后请在 <answer> ... </answer> 标签中给出最终答案。在 <think> 部分，务必进行逐步思考、结构分析以及对错误选项的排除，要求如下：

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
仅写最终答案，无需重复分析

现在请根据以上要求，在 <think> 中展示详细推理，在 <answer> 中填写最终答案。请使用中文回复"""
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -----------------------------
# 6. 推理生成
# -----------------------------
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,    # 控制生成的 token 长度
        do_sample=True,       # 采用采样策略
        top_k=40,             # 限制解码时的候选 token 范围
        top_p=0.9,            # nucleus sampling
        temperature=0.3    # 生成的发散程度
    )

# -----------------------------
# 7. 解码输出结果
# -----------------------------
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("生成结果：", result)
