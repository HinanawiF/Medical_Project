#!/bin/bash

# 基于RJUA-QA数据集微调医疗大模型的脚本

# 设置环境变量
export HF_HOME=./hf_cache
export TRANSFORMERS_CACHE=./hf_cache

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 下载RJUA-QA数据集
if [ ! -d "RJUA-QA" ]; then
    echo "下载RJUA-QA数据集..."
    # 注意：这里需要替换为实际的数据集下载链接
    wget https://example.com/RJUA-QA.zip
    unzip RJUA-QA.zip
fi

# 微调Llama模型
echo "开始微调Llama模型..."
python medical_llm_finetune.py \
    --model_type llama \
    --dataset_path RJUA-QA \
    --output_dir outputs/llama-rjua-qa \
    --lora_rank 8 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --max_seq_length 1024 \
    --do_train \
    --do_eval

# 微调DeepSeek模型
echo "开始微调DeepSeek模型..."
python medical_llm_finetune.py \
    --model_type deepseek \
    --dataset_path RJUA-QA \
    --output_dir outputs/deepseek-rjua-qa \
    --lora_rank 8 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --max_seq_length 1024 \
    --do_train \
    --do_eval

# 微调Qwen模型
echo "开始微调Qwen模型..."
python medical_llm_finetune.py \
    --model_type qwen \
    --dataset_path RJUA-QA \
    --output_dir outputs/qwen-rjua-qa \
    --lora_rank 8 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --max_seq_length 1024 \
    --do_train \
    --do_eval

echo "所有模型微调完成！"