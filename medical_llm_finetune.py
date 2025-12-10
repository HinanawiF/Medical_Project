import os
import argparse
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from trl import SFTTrainer
from unsloth import LoRAConfig, is_bfloat16_supported
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from evaluate import load
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保中文分词正常工作
try:
    import nltk
    nltk.data.path.append("/tmp/nltk_data")
    nltk.download('punkt', quiet=True, download_dir="/tmp/nltk_data")
except Exception as e:
    logger.warning(f"下载nltk punkt失败: {e}，可能影响中文分词效果")

# 支持的模型列表
SUPPORTED_MODELS = {
    "llama": {
        "name": "meta-llama/Llama-2-7b-hf",
        "tokenizer_kwargs": {"trust_remote_code": True},
        "model_kwargs": {"trust_remote_code": True}
    },
    "deepseek": {
        "name": "deepseek-ai/deepseek-coder-1.3b-base",
        "tokenizer_kwargs": {"trust_remote_code": True},
        "model_kwargs": {"trust_remote_code": True}
    },
    "qwen": {
        "name": "Qwen/Qwen-7B",
        "tokenizer_kwargs": {"trust_remote_code": True},
        "model_kwargs": {"trust_remote_code": True}
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="基于RJUA-QA数据集微调医疗大模型")
    parser.add_argument("--model_type", type=str, default="llama", 
                        choices=SUPPORTED_MODELS.keys(), help="选择要微调的模型类型")
    parser.add_argument("--dataset_path", type=str, default="RJUA-QA", 
                        help="RJUA-QA数据集路径或名称")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="微调模型的输出目录")
    parser.add_argument("--lora_rank", type=int, default=8, 
                        help="LoRA适配器的秩")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                        help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, 
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="梯度累积步数")
    parser.add_argument("--max_steps", type=int, default=1000, 
                        help="最大训练步数")
    parser.add_argument("--max_seq_length", type=int, default=1024, 
                        help="最大序列长度")
    parser.add_argument("--eval_steps", type=int, default=50, 
                        help="评估频率")
    parser.add_argument("--save_steps", type=int, default=100, 
                        help="保存模型频率")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="是否使用4位量化加载模型")
    parser.add_argument("--do_train", action="store_true", 
                        help="是否进行训练")
    parser.add_argument("--do_eval", action="store_true", 
                        help="是否进行评估")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="从检查点恢复训练")
    return parser.parse_args()

def load_rjua_qa_dataset(dataset_path, split="train"):
    """加载RJUA-QA医疗数据集"""
    try:
        # 尝试从Hugging Face Hub加载
        dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        logger.warning(f"无法从Hugging Face Hub加载数据集: {e}，尝试从本地加载")
        # 从本地加载
        if not os.path.exists(dataset_path):
            raise ValueError(f"数据集路径不存在: {dataset_path}")
        
        # 假设RJUA-QA数据集是JSONL格式
        data_files = {
            "train": os.path.join(dataset_path, "train.jsonl"),
            "validation": os.path.join(dataset_path, "validation.jsonl"),
            "test": os.path.join(dataset_path, "test.jsonl")
        }
        
        if not os.path.exists(data_files[split]):
            raise ValueError(f"找不到{split}分割的数据文件: {data_files[split]}")
        
        dataset = load_dataset("json", data_files=data_files, split=split)
    
    logger.info(f"加载{split}数据集成功，样本数: {len(dataset)}")
    return dataset

def preprocess_rjua_qa(examples, tokenizer, max_length=1024):
    """预处理RJUA-QA数据集，转换为模型可接受的格式"""
    prompts = []
    
    for question, answer in zip(examples["question"], examples["answer"]):
        # 构建医疗指令模板
        prompt = f"""### 医疗问题:
{question}

### 专业回答:
{answer}"""
        prompts.append(prompt)
    
    # 编码文本
    model_inputs = tokenizer(prompts, max_length=max_length, truncation=True)
    
    # 准备标签（与输入相同，因为我们在做生成任务）
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def create_model_and_tokenizer(model_type, load_in_4bit=False):
    """创建模型和分词器"""
    model_config = SUPPORTED_MODELS[model_type]
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        **model_config["tokenizer_kwargs"]
    )
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        **model_config["model_kwargs"]
    )
    
    # 准备模型进行4位训练
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    logger.info(f"成功加载{model_type}模型和分词器")
    return model, tokenizer

def setup_lora_config(rank=8):
    """设置LoRA配置"""
    return LoRAConfig(
        r=rank,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """训练模型"""
    # 设置LoRA配置
    lora_config = setup_lora_config(args.lora_rank)
    
    # 包装模型
    if not args.load_in_4bit:
        model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    # 设置训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=args.output_dir,
        report_to="none",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset is not None else False,
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        args=training_args,
        peft_config=lora_config if args.load_in_4bit else None,
    )
    
    # 开始训练
    logger.info("开始训练模型...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 保存模型
    logger.info(f"训练完成，保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    return model

def evaluate_model(model, tokenizer, eval_dataset, args):
    """评估模型性能"""
    logger.info("开始评估模型...")
    
    # 加载评估指标
    rouge = Rouge()
    bleu_smoothing = SmoothingFunction().method4
    
    # 准备评估结果列表
    results = []
    
    # 为每个样本生成回答并评估
    for example in eval_dataset:
        question = example["question"]
        reference_answer = example["answer"]
        
        # 构建输入提示
        prompt = f"""### 医疗问题:
{question}

### 专业回答:"""
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=args.max_seq_length,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                num_return_sequences=1
            )
        
        # 解码生成的回答
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除提示部分
        generated_answer = generated_answer.replace(prompt, "").strip()
        
        # 计算评估指标
        try:
            # ROUGE分数
            rouge_scores = rouge.get_scores(generated_answer, reference_answer)[0]
            
            # BLEU分数 (处理中文需要分词)
            # 简单的中文字符分词
            gen_tokens = [char for char in generated_answer]
            ref_tokens = [char for char in reference_answer]
            
            bleu_score = sentence_bleu(
                [ref_tokens], 
                gen_tokens,
                smoothing_function=bleu_smoothing
            )
            
            # 保存结果
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"],
                "bleu": bleu_score
            })
        except Exception as e:
            logger.error(f"评估样本时出错: {e}")
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge-1": 0,
                "rouge-2": 0,
                "rouge-l": 0,
                "bleu": 0
            })
    
    # 计算平均分数
    avg_rouge_1 = np.mean([r["rouge-1"] for r in results])
    avg_rouge_2 = np.mean([r["rouge-2"] for r in results])
    avg_rouge_l = np.mean([r["rouge-l"] for r in results])
    avg_bleu = np.mean([r["bleu"] for r in results])
    
    # 打印评估结果
    logger.info(f"评估结果:")
    logger.info(f"  ROUGE-1: {avg_rouge_1:.4f}")
    logger.info(f"  ROUGE-2: {avg_rouge_2:.4f}")
    logger.info(f"  ROUGE-L: {avg_rouge_l:.4f}")
    logger.info(f"  BLEU: {avg_bleu:.4f}")
    
    # 保存详细结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细评估结果已保存到 {results_file}")
    
    return {
        "rouge-1": avg_rouge_1,
        "rouge-2": avg_rouge_2,
        "rouge-l": avg_rouge_l,
        "bleu": avg_bleu
    }

def main():
    args = parse_args()
    
    # 检查输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 记录参数
    params_file = os.path.join(args.output_dir, "training_params.json")
    with open(params_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer(args.model_type, args.load_in_4bit)
    
    # 加载数据集
    if args.do_train:
        train_dataset = load_rjua_qa_dataset(args.dataset_path, split="train")
        eval_dataset = load_rjua_qa_dataset(args.dataset_path, split="validation")
        
        # 预处理数据集
        train_dataset = train_dataset.map(
            lambda x: preprocess_rjua_qa(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_rjua_qa(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # 训练模型
        model = train_model(model, tokenizer, train_dataset, eval_dataset, args)
    
    # 评估模型
    if args.do_eval:
        # 如果没有训练，加载微调后的模型
        if not args.do_train:
            model = PeftModel.from_pretrained(
                model, 
                args.output_dir if args.resume_from_checkpoint is None else args.resume_from_checkpoint
            )
        
        eval_dataset = load_rjua_qa_dataset(args.dataset_path, split="test")
        
        # 评估模型
        evaluate_model(model, tokenizer, eval_dataset, args)

if __name__ == "__main__":
    main()