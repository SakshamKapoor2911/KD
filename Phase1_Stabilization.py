# Enable autoreload to automatically pick up modifications in your core/ modules
# %load_ext autoreload
# %autoreload 2


import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Import our clean modularized classes
from core.models import ExpertFeedbackModel, AmateurFeedbackModel, ParsingModel
from core.network import AmateurExpertFeedbackNetWork
from core.evaluate import RefinedAnswerEvaluator, comparative_testing_results_log, _is_correct

class ResourceTracker:
    def __init__(self, model_params=1.1e9):  # TinyLlama has ~1.1B parameters
        self.model_params = model_params

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_vram = torch.cuda.memory_allocated()
        else:
            self.start_vram = 0
        self.start_time = time.time()

    def stop(self, generated_tokens):
        end_time = time.time()
        latency = end_time - self.start_time
        
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)  # In MB
        else:
            peak_vram = 0.0
            
        # Estimate FLOPs: 2 * params * generated_tokens per forward pass
        flops_estimate = 2 * self.model_params * generated_tokens
        
        return {
            "latency_seconds": latency,
            "peak_vram_mb": peak_vram,
            "estimated_flops": flops_estimate
        }

teacher_model_name = "meta-llama/Llama-3.1-8B-Instruct"
student_model_name = "meta-llama/Llama-3.2-1B-Instruct"

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

expert_model = ExpertFeedbackModel(teacher_tokenizer, teacher_model, teacher_model_name)
amateur_feedback_model = AmateurFeedbackModel(student_tokenizer, student_model, student_model_name)
parse_model = ParsingModel(teacher_tokenizer, teacher_model, teacher_model_name)

def evaluate_amateur_only(question, tracker, is_math=False, examples=None):
    tracker.start()
    
    # 1. Generate Initial Answer
    initial_answer = amateur_feedback_model.generate_answer(question, is_math=is_math, examples=examples)
    
    # 2. Generate Amateur Feedback & Score
    feedback, score, _ = amateur_feedback_model.generate_feedback(question, initial_answer, is_math=is_math, examples=examples)
    
    # 3. Apply Feedback to produce Improved Answer
    improved_answer = amateur_feedback_model.apply_feedback(question, initial_answer, feedback)
    
    # 4. Generate Self-Critique
    self_critique = amateur_feedback_model.generate_self_critique(question, improved_answer)
    
    # 5. Apply Critique for final refined answer
    final_answer = amateur_feedback_model.apply_self_critique(question, improved_answer, self_critique)
    
    # Tokens produced estimation (rough length in characters / 4)
    total_chars = len(initial_answer) + len(feedback) + len(improved_answer) + len(self_critique) + len(final_answer)
    approx_tokens = total_chars // 4
    
    metrics = tracker.stop(approx_tokens)
    
    return {
        "initial_answer": initial_answer,
        "feedback": feedback,
        "improved_answer": improved_answer,
        "self_critique": self_critique,
        "final_answer": final_answer,
        "metrics": metrics
    }

tracker = ResourceTracker(model_params=1.1e9)
sample_question = "A train travels at 80 km/h for 3 hours. How far does it travel?"

print("Running Amateur-Only self-critique loop...")
result = evaluate_amateur_only(sample_question, tracker, is_math=True)

print("\n--- INITIAL ANSWER ---")
print(result["initial_answer"])

print("\n--- AMATEUR FEEDBACK ---")
print(result["feedback"])

print("\n--- REFINED ANSWER ---")
print(result["final_answer"])

print("\n--- COST & RESOURCE METRICS ---")
for k, v in result["metrics"].items():
    if "flops" in k:
        print(f"{k}: {v:.2e} FLOPs")
    else:
        print(f"{k}: {v:.2f}")

# Set loss flags to ONLY optimize LM Loss (Ablate Hidden, Scoring, and Logit losses)
# [lm_loss, hidden_loss, scoring_loss, logit_loss]
loss_flags = [True, False, False, False]

# load dummy dataset of 3 samples from JSON
try:
    with open("300_sample.jsonl", "r") as f:
        import json
        expert_datasets = [json.loads(line) for line in f][:5]
except FileNotFoundError:
    # Fallback dummy sample if dataset is not found in folder
    expert_datasets = [
        {
            "prompt": "Solve 2x + 5 = 15",
            "answer": "2x = 10 -> x = 5",
            "score": 1.0,
            "feedback": "Excellent step-by-step resolution."
        }
    ]

network = AmateurExpertFeedbackNetWork(
    student=amateur_feedback_model,
    teacher=expert_model,
    expert_datasets=expert_datasets,
    loss_flags=loss_flags
)

print("Active losses in training:", network.loss_config.get_active_losses()[0])

print("Starting Ablated Knowledge Distillation Training...")
# Run for 1 epoch, stopped after 2 iterations to demonstrate stability
weights = [1.0] # Only one active loss weight (LM Loss)
kd_result = network.knowledge_distillation_with_SFT(weights=weights, batch_size=1, epochs=1, stopped=2)
print("Training done!")
