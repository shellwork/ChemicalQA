import openai
import concurrent.futures
import itertools
import time
import json

# 多个 ModelScope API Keys
API_KEYS = [
    "",
    ""
]

BASE_URL = "https://api-inference.modelscope.cn/v1/"
MODEL_ID = "deepseek-ai/DeepSeek-R1"  # 使用合适的 Model Id

# 轮询 API Keys
api_key_cycle = itertools.cycle(API_KEYS)

def generate_prompt(data):
    """根据数据生成 Prompt（中文）"""
    prompt = f"""
    你是一名化学专家，需要根据提供的分子结构（SMILES）、问题（question）及正确选项（answer），按以下步骤输出选择正确答案的思考过程：  

    ---

    ### **步骤1：结构解析**  
    1. **核心骨架识别**：  
       - 确定分子核心（如四环素、苯环、杂环、β-内酰胺环等）。  
       - 标注立体化学特征（手性中心`@`/`@@`、顺反异构）。  
    2. **关键官能团标注**：  
       - 列出所有功能基团（羟基-OH、氨基-NH2、酯基-COO、卤素-Cl等）。  
       - 标记影响性质的关键基团（如亲脂性基团、氢键供体/受体）。  

    ### **步骤2：物化性质与功能关联**  
    1. **性质推断**：  
       - **极性/亲脂性**：根据基团计算logP趋势（如二甲氨基↑logP，羧酸↓logP）。  
       - **溶解度**：离子化基团（-COOH、-NH2）增强水溶性，疏水基团（芳环、烷基）降低溶解度。  
       - **稳定性**：识别易水解（酯、β-内酰胺）、易氧化（酚羟基）的基团。  
    2. **关联问题关键词**：  
       - 若问题涉及**吸收**，重点分析logP、离子化状态、膜渗透性。  
       - 若涉及**代谢**，关注易酶解基团（如酯键）。  

    ### **步骤3：选项分析与排除**  
    1. **正确选项**：  
       - 明确结构特征如何支持答案（例如：“酯键易被肠壁酯酶水解 → 选项B”）。  
    2. **错误选项**：  
       - 逐条反驳（例如：“选项D声称100%生物利用度，但羧酸在肠道电离，阻碍吸收”）。  

    ### **步骤4：结论**  
    - 总结1-2个核心结构特征，解释其如何排除其他选项并唯一支持正确答案。  

    ---

    ### **格式要求**  
    - 分步骤回答，使用`**加粗**`标注关键术语（如**酯键**、**logP**）。  
    - 避免冗长，每个步骤不超过3个要点。  
    - 若分子属于某类药物（如β-内酰胺抗生素），需明确说明（如“β-内酰胺环提示抗生素活性”）。  
    
    分子表示法（SMILES）：{data['smiles']}
    
    下面是你的问题：{data['question']}
    这是该问题的正确答案：{data['answer']}
    
    请详细阐述你的推理过程。
    """
    return prompt

def call_api(prompt, api_key):
    """单次 API 请求"""
    try:
        client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "你是一个专业的化学分析助手。"},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.6
        )
        
        result = ""
        for chunk in response:
            if chunk.choices:
                result += chunk.choices[0].delta.content
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def process_requests(data_list, output_file, max_workers=20):
    """多线程并发请求 API，实时更新 JSONL 文件"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, open(output_file, "a", encoding="utf-8") as f:
        future_to_data = {
            executor.submit(call_api, generate_prompt(data), next(api_key_cycle)): data
            for data in data_list
        }
        
        for future in concurrent.futures.as_completed(future_to_data):
            data = future_to_data[future]
            try:
                reasoning_content = future.result()
                result_entry = {
                    "cid": data["cid"],
                    "input": data["question"],
                    "content": data["answer"],
                    "reasoning_content": reasoning_content,
                    "smiles": data["smiles"],
                    "id": data["id"]
                }
            except Exception as e:
                result_entry = {
                    "cid": data["cid"],
                    "input": data["question"],
                    "content": data["answer"],
                    "reasoning_content": f"Error: {str(e)}",
                    "smiles": data["smiles"],
                    "id": data["id"]
                }
            
            # 实时写入 JSONL 文件
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            f.flush()
            
            # 在控制台打印部分信息确认 API 正常运行
            print(f"已处理问题: {data['question'][:30]}...，部分答案: {reasoning_content[:50]}...")

def load_json(file_path):
    """从 JSON 文件加载数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    input_file = "question/test1.json"  # 请输入你的 JSON 数据文件路径
    output_file = "question/answer1.jsonl"
    dataset = load_json(input_file)
    
    start_time = time.time()
    process_requests(dataset, output_file, max_workers=20)
    end_time = time.time()
    
    print(f"已处理 {len(dataset)} 条数据，共耗时 {end_time - start_time:.2f} 秒。")
