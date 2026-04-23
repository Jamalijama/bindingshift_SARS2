import pandas as pd
import os
import string
import random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 1. 模型和分词器加载
model_name = "./esm2_t33_650M_UR50D"  # 可选其他规模：t12_35M, t30_150M, t33_650M等
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 2. 数据准备（模拟蛋白序列，无需外部文件）
# 生成100条模拟蛋白序列（长度50-200不等，仅含标准氨基酸字母）

# 标准20种氨基酸字母表
amino_acids = "ACDEFGHIKLMNPQRSTVWY"


# def generate_random_sequence(min_len=200, max_len=500):
#     length = random.randint(min_len, max_len)
#     return ''.join(random.choice(amino_acids) for _ in range(length))


# load_dataset

path = './data/'
df = pd.DataFrame()

for file in os.listdir(path):
    if file.endswith('df_parsed_SpikeS1_MERS_SARS.csv') & file.startswith('df_parsed_SpikeS1_MERS_SARS.csv'):
        df = pd.read_csv(path + file, index_col='seqID', encoding='gbk')

sequences = df.SpikeS1.tolist()
num_sequences = len(sequences)

dataset = Dataset.from_dict({"sequence": sequences})
# print(sequences)

# 3. 数据预处理函数


def preprocess_function(examples):
    # 分词（添加特殊 tokens，如<cls>和<<eos>）
    return tokenizer(
        examples["sequence"],
        padding="max_length",
        truncation=False,
        max_length=300,  # 根据模型支持的最大长度调整
        return_tensors="pt"
    )


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["sequence"]  # 移除原始序列列
)

# 4. 数据collator（用于MLM任务的掩码处理）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # 启用掩码语言模型任务
    mlm_probability=0.15  # 随机掩码15%的token
)

# 5. 训练参数设置
training_args = TrainingArguments(
    output_dir="./esm2_finetuned_MERSSARSS1",  # 模型保存路径
    overwrite_output_dir=True,
    num_train_epochs=200,  # 模拟数据量小， epochs可设小些
    per_device_train_batch_size=20,
    gradient_accumulation_steps=10,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=10,  # 模拟数据少，日志步长设小些
    save_steps=100,
    fp16=True,  # 启用混合精度训练（需GPU支持）
    report_to="none"
)

# 6. 初始化Trainer并微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 开始微调
trainer.train()

# 7. 保存微调后的模型和分词器
model.save_pretrained("./esm2_finetuned_MERSSARSS1")
tokenizer.save_pretrained("./esm2_finetuned_MERSSARSS1")


# 8. 从微调后的模型提取序列Embedding
def get_sequence_embedding(sequence, model, tokenizer, max_length=512):
    """输入单个蛋白序列，返回其Embedding（使用<cls> token的输出）"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(model.device)  # 移动到模型所在设备（GPU/CPU）

        outputs = model(**inputs, output_hidden_states=True)
        # 取最后一层的<cls> token输出作为序列Embedding
        cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze()
    return cls_embedding.cpu().numpy()  # 转为numpy数组


# 示例：提取单个测试序列的Embedding
# df_test = pd.read_csv ('./data/df_spikeprot_S1dedup_S1high95_Shigh95_132204_v2_testing32204.csv', encoding = 'gbk')
# test_sequence = df_test['prS1'].tolist()
# embedding = get_sequence_embedding(
#     test_sequence,
#     model=model,
#     tokenizer=tokenizer
# )
# print(f"测试序列: {test_sequence[:20]}...（长度100）")
# print(f"序列Embedding形状: {embedding.shape}")  # 输出应为(384,)（对应8M模型）
