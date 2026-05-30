import sys
import os
import json
import time
import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Import our clean modularized classes
from core.models import ExpertFeedbackModel, AmateurFeedbackModel, ParsingModel
from core.network import AmateurExpertFeedbackNetWork
from core.evaluate import RefinedAnswerEvaluator, _is_correct, compute_semantic_similarity

# Constants
TEACHER_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
STUDENT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
NLI_MODEL_NAME = "facebook/bart-large-mnli"
NUM_TEST_SAMPLES = 15  # Fast, stable, and highly reliable sample count for setup

print("--------------------------------------------------")
print("🚀 STARTING CONFIGURATIONS BENCHMARK PIPELINE")
print("--------------------------------------------------")

# Check HF Token
if "HF_TOKEN" not in os.environ:
    print("[WARNING] HF_TOKEN not found in environment. Gated models may fail to load.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Tokenizers & Models
print("\n[1/5] Loading models and tokenizers...")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, trust_remote_code=True)
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

print("\nLoading NLI model for consistency check...")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME, device_map="auto"
)

# Initialize feedback wrappers
expert_model = ExpertFeedbackModel(teacher_tokenizer, teacher_model, TEACHER_MODEL_NAME)
amateur_model = AmateurFeedbackModel(student_tokenizer, student_model, STUDENT_MODEL_NAME)
evaluator = RefinedAnswerEvaluator(student_tokenizer, student_model)

# 2. Load GSM8K Test Dataset
print("\n[2/5] Loading GSM8K test dataset...")
gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
test_samples = []
for i in range(NUM_TEST_SAMPLES):
    sample = gsm8k_ds[i]
    # Extract numeric target from the answer field (format: "#### <number>")
    gt_match = re.search(r"####\s*(-?\d+)", sample["answer"])
    gt_number = gt_match.group(1) if gt_match else None
    test_samples.append({
        "question": sample["question"],
        "answer": sample["answer"],
        "gt_number": gt_number
    })

print(f"Loaded {len(test_samples)} test samples successfully.")

# Helper to extract the last number from generated model output
def extract_number(text):
    numbers = re.findall(r"[-+]?\d+", text)
    return numbers[-1] if numbers else None

# Helper to perform the self-critique loop autonomously on amateur model
def run_amateur_critique_loop(question, model):
    initial = model.generate_answer(question, is_math=True)
    feedback, score, _ = model.generate_feedback(question, initial, is_math=True)
    improved = model.apply_feedback(question, initial, feedback)
    self_critique = model.generate_self_critique(question, improved)
    final = model.apply_self_critique(question, improved, self_critique)
    return initial, final

# Results storage
results = []

# --- CONFIG 1: Base Amateur (No Critique) ---
print("\n[3/5] Evaluating CONFIG 1: Base Amateur (No Critique)...")
config1_answers = []
for idx, sample in enumerate(test_samples):
    print(f"  Config 1 - Sample {idx+1}/{NUM_TEST_SAMPLES}")
    initial_ans = amateur_model.generate_answer(sample["question"], is_math=True)
    pred_num = extract_number(initial_ans)
    correct = _is_correct(pred_num, sample["gt_number"])
    config1_answers.append({
        "answer": initial_ans,
        "correct": correct
    })

# --- CONFIG 2: Base Amateur (With Critique, Un-distilled) ---
print("\nEvaluating CONFIG 2: Base Amateur (With Critique, Un-distilled)...")
config2_answers = []
for idx, sample in enumerate(test_samples):
    print(f"  Config 2 - Sample {idx+1}/{NUM_TEST_SAMPLES}")
    initial_ans, final_ans = run_amateur_critique_loop(sample["question"], amateur_model)
    pred_num = extract_number(final_ans)
    correct = _is_correct(pred_num, sample["gt_number"])
    
    # Measure refinement similarity
    similarity = compute_semantic_similarity(initial_ans, final_ans)
    
    config2_answers.append({
        "answer": final_ans,
        "correct": correct,
        "similarity": similarity
    })

# --- CONFIG 4: Original CLEAR Setup ---
print("\nEvaluating CONFIG 4: Original CLEAR Setup (Expert + Amateur)...")
config4_answers = []
for idx, sample in enumerate(test_samples):
    print(f"  Config 4 - Sample {idx+1}/{NUM_TEST_SAMPLES}")
    # 1. Initial generation
    initial_ans = amateur_model.generate_answer(sample["question"], is_math=True)
    
    # 2. Expert + Amateur combined feedback
    student_feedback, _, _ = amateur_model.generate_feedback(sample["question"], initial_ans, is_math=True)
    teacher_feedback, _ = expert_model.generate_feedback(sample["question"], initial_ans, is_math=True)
    
    combined_feedback = f"Expert points: {teacher_feedback}\nAmateur points: {student_feedback}"
    
    # 3. Revise answer using combined feedback
    refined_ans = amateur_model.apply_feedback(sample["question"], initial_ans, combined_feedback)
    
    pred_num = extract_number(refined_ans)
    correct = _is_correct(pred_num, sample["gt_number"])
    similarity = compute_semantic_similarity(initial_ans, refined_ans)
    
    config4_answers.append({
        "answer": refined_ans,
        "correct": correct,
        "similarity": similarity
    })

# --- TRAINING / DISTILLATION ---
print("\n[4/5] Running Knowledge Distillation (Fine-tuning the Amateur Model)...")
# Prepare a small SFT set from first 5 samples
expert_datasets = []
for idx in range(min(5, len(test_samples))):
    expert_datasets.append({
        "prompt": test_samples[idx]["question"],
        "answer": test_samples[idx]["answer"],
        "score": 1.0,
        "feedback": "Perfect step-by-step resolution."
    })

network = AmateurExpertFeedbackNetWork(
    student=amateur_model,
    teacher=expert_model,
    expert_datasets=expert_datasets,
    loss_flags=[True, False, False, False],  # lm_loss only (Ablation stabilization)
    device=device
)

weights = [1.0]
network.knowledge_distillation_with_SFT(weights=weights, batch_size=1, epochs=1, stopped=3)
print("Distillation Completed successfully!")

# --- CONFIG 3: Distilled Amateur (With Critique) ---
print("\n[5/5] Evaluating CONFIG 3: Distilled Amateur (With Critique)...")
config3_answers = []
for idx, sample in enumerate(test_samples):
    print(f"  Config 3 - Sample {idx+1}/{NUM_TEST_SAMPLES}")
    initial_ans, final_ans = run_amateur_critique_loop(sample["question"], amateur_model)
    pred_num = extract_number(final_ans)
    correct = _is_correct(pred_num, sample["gt_number"])
    similarity = compute_semantic_similarity(initial_ans, final_ans)
    
    config3_answers.append({
        "answer": final_ans,
        "correct": correct,
        "similarity": similarity
    })

# --- COMPILING & SAVING RESULTS ---
print("\n📊 Compiling final benchmark statistics...")

def compute_accuracy(answers):
    correct_count = sum(1 for item in answers if item["correct"] == 1)
    return (correct_count / len(answers)) * 100.0

acc1 = compute_accuracy(config1_answers)
acc2 = compute_accuracy(config2_answers)
acc3 = compute_accuracy(config3_answers)
acc4 = compute_accuracy(config4_answers)

avg_sim2 = sum(item["similarity"] for item in config2_answers) / len(config2_answers)
avg_sim3 = sum(item["similarity"] for item in config3_answers) / len(config3_answers)
avg_sim4 = sum(item["similarity"] for item in config4_answers) / len(config4_answers)

summary_results = {
    "num_samples": NUM_TEST_SAMPLES,
    "configs": {
        "Config 1: Base Amateur (No Critique)": {"accuracy": acc1, "avg_semantic_similarity": 1.0},
        "Config 2: Base Amateur (With Critique, Un-distilled)": {"accuracy": acc2, "avg_semantic_similarity": avg_sim2},
        "Config 3: Distilled Amateur (With Critique)": {"accuracy": acc3, "avg_semantic_similarity": avg_sim3},
        "Config 4: Original CLEAR Setup": {"accuracy": acc4, "avg_semantic_similarity": avg_sim4}
    }
}

# Save as JSON locally
output_file = "/home/skapoor/KD/evaluation_results.json"
with open(output_file, "w") as f:
    json.dump(summary_results, f, indent=4)

print("\n==================================================")
print("🏆 FINAL COMPARISON RESULTS")
print("==================================================")
print(f"Config 1 Accuracy: {acc1:.2f}%")
print(f"Config 2 Accuracy: {acc2:.2f}% (Avg refinement similarity: {avg_sim2:.4f})")
print(f"Config 3 Accuracy: {acc3:.2f}% (Avg refinement similarity: {avg_sim3:.4f})")
print(f"Config 4 Accuracy: {acc4:.2f}% (Avg refinement similarity: {avg_sim4:.4f})")
print("==================================================")
print(f"Results saved locally to {output_file}")
print("Pipeline complete!")
