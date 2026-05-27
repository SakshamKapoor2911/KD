import torch
import torch.nn as nn
from torch.optim import AdamW
import random

from core.utils import LossConfig, AdaptiveWeightedKDPolicyEMA, visualize_student_performance
from core.losses import (
    compute_lm_loss,
    compute_hidden_loss,
    compute_scoring_loss,
    compute_logit_standardization,
    compute_all_losses_vectorized,
    unpack_feedback_and_scores
)

class AmateurExpertFeedbackNetWork:
    def __init__(self, student, teacher, expert_datasets=None, loss_flags=None, device=None):
        self.student = student
        self.teacher = teacher
        self.expert_datasets = expert_datasets or []
        self.student_tokenizer = student.tokenizer
        self.teacher_tokenizer = teacher.tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student.score_head = self.student.score_head.to(torch.float32)
        self.model_dtype = torch.float32
        self.loss_flags = [True, True, True, True] if loss_flags is None else loss_flags
        self.loss_config = LossConfig(self.loss_flags)

        self.threshold_policy = AdaptiveWeightedKDPolicyEMA(
            num_losses=self.loss_config.num_enabled, device=self.device
        )

        for tokenizer in [self.student_tokenizer, self.teacher_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        self.teacher.model.eval()
        with torch.no_grad():
            dummy_text = "Hello World"
            student_input = self.student_tokenizer(dummy_text, return_tensors="pt").to(self.device)
            teacher_input = self.teacher_tokenizer(dummy_text, return_tensors="pt").to(self.device)

            student_out = self.student.model(**student_input, output_hidden_states=True, return_dict=True)
            teacher_out = self.teacher.model(**teacher_input, output_hidden_states=True, return_dict=True)

            student_hidden_dim = student_out.hidden_states[-1].size(-1)
            teacher_hidden_dim = teacher_out.hidden_states[-1].size(-1)

        if student_hidden_dim != teacher_hidden_dim:
            self.align_hidden_for_hidden_loss = nn.Linear(student_hidden_dim, teacher_hidden_dim)
        else:
            self.align_hidden_for_hidden_loss = nn.Identity()

        self.align_hidden_for_hidden_loss = self.align_hidden_for_hidden_loss.to(self.device).to(self.model_dtype)

        same_vocab = (
            self.student.model.config.vocab_size == self.teacher.model.config.vocab_size
            and self.student_tokenizer.get_vocab() == self.teacher_tokenizer.get_vocab()
        )

        self.projection_layer = nn.Identity()

        if not same_vocab:
            print("[Logit KD] Different tokenizers/vocabs detected → disabling logit_loss.")
            try:
                self.loss_config.toggle_loss("logit_loss", False)
            except Exception:
                pass

        self.optimizer = AdamW(
            [
                {"params" : self.student.model.parameters(), "lr" : 5e-7, "weight_decay" : 0.01},
                {"params": self.student.score_head.parameters(), "lr": 5e-5, "weight_decay": 0.01},
                {"params": self.align_hidden_for_hidden_loss.parameters(), "lr": 1e-6, "weight_decay": 0.01},
                {"params": self.projection_layer.parameters(), "lr": 1e-6, "weight_decay": 0.01},
            ],
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def knowledge_distillation_with_SFT(self, weights, batch_size=3, epochs=3, stopped=20, is_math=False, examples=None):
        random.shuffle(self.expert_datasets)

        loss_names, _ = self.loss_config.get_active_losses()
        loss_history = {loss_name: [] for loss_name in loss_names}

        total_steps = epochs * min(stopped, len(self.expert_datasets))
        global_step = 0

        epoch_avgs = {}

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

            epoch_losses = {name: 0.0 for name in loss_names}
            epoch_batch_counts = {name: 0 for name in loss_names}

            num_samples = 0
            stop_epoch = False
            batches_in_epoch = 0

            last_question_and_answer_in_batch = ""
            for batch_start in range(0, len(self.expert_datasets), batch_size):
                batch = self.expert_datasets[batch_start: batch_start + batch_size]
                if stopped is not None and num_samples >= stopped:
                    break

                each_loss_in_batch = {loss_name: [] for loss_name in loss_names}

                for sample in batch:
                    prompt_text = sample.get("prompt")
                    answer = sample.get("answer")
                    expert_score = sample.get("score")
                    expert_feedback = sample.get("feedback")

                    if not prompt_text or not answer or expert_score is None:
                        continue

                    print(f"\nSample {num_samples + 1}")
                    student_feedback, student_score_value, student_score_logit = self.student.generate_feedback(
                        prompt_text, answer, is_math, examples
                    )
                    with torch.no_grad():
                        teacher_feedback, teacher_score = self.teacher.generate_feedback(
                            prompt_text, answer, is_math, examples
                        )

                    if "lm_loss" in loss_names:
                        lm_loss = compute_lm_loss(teacher_feedback, self.student, prompt_text, answer,
                                                  self.device, self.model_dtype, is_math, examples)
                        each_loss_in_batch["lm_loss"].append(lm_loss)

                    if "hidden_loss" in loss_names:
                        hidden_loss = compute_hidden_loss(teacher_feedback, self.student_tokenizer, self.teacher_tokenizer,
                                                          self.student, self.teacher, self.align_hidden_for_hidden_loss,
                                                          self.device, self.model_dtype)
                        each_loss_in_batch["hidden_loss"].append(hidden_loss)

                    if "scoring_loss" in loss_names:
                        score_loss = compute_scoring_loss(student_score_logit, teacher_score,
                                                          self.model_dtype, self.device)
                        each_loss_in_batch["scoring_loss"].append(score_loss)

                    if "logit_loss" in loss_names:
                        logit_loss = compute_logit_standardization(self.student, self.teacher,
                                                                  self.student_tokenizer, self.teacher_tokenizer,
                                                                  self.projection_layer, expert_feedback)
                        each_loss_in_batch["logit_loss"].append(logit_loss)

                    num_samples += 1
                    last_question_and_answer_in_batch = prompt_text + answer

                batch_losses = {
                    name: torch.stack(vals).mean()
                    for name, vals in each_loss_in_batch.items() if len(vals) > 0
                }
                if not batch_losses:
                    continue

                curr_losses_detached = [batch_losses[name].detach() for name in loss_names if name in batch_losses]

                if global_step == 0:
                    assigned_weights = weights
                    skip_kd = False
                    if_freeze_student = False
                else:
                    assigned_weights, skip_kd, if_freeze_student = self.threshold_policy.update_weights(
                        curr_losses_detached, last_question_and_answer_in_batch
                    )

                loss_weight_map = dict(zip(loss_names, assigned_weights))
                weights_str = ", ".join([f"{name}={w:.4f}" for name, w in loss_weight_map.items()])
                print(f"[Adaptive Weights] {weights_str}, skip_kd={skip_kd}")

                total_loss = 0.0
                for name, w in loss_weight_map.items():
                    if name in batch_losses and batch_losses[name] is not None:
                        total_loss = total_loss + w * self.loss_config.scales.get(name, 1.0) * batch_losses[name]

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                for name, loss_val in batch_losses.items():
                    epoch_losses[name] += float(loss_val.item())
                    epoch_batch_counts[name] += 1

                for name, loss_val in batch_losses.items():
                    loss_history[name].append(float(loss_val.item()))

                print("---- FEEDBACK COMPARISON ----")
                print(f"[Student Feedback]: {student_feedback}")
                print(f"[Teacher Feedback]: {teacher_feedback}")
                print(f"[Expert  Feedback]: {expert_feedback}")
                print("---- LOSS VALUES ----")
                for name, batch_loss in batch_losses.items():
                    print(f"{name}: {batch_loss.item():.4f}")
                print(f"TOTAL LOSS: {total_loss.item():.4f}")
                print("------------------------------\n")

                global_step += 1
                batches_in_epoch += 1

                if skip_kd:
                    print("---- SKIPPING KD ----")
                    print(f"Epoch: {epoch + 1}/{epochs}, Iteration: {batch_start // batch_size + 1}, Global step: {global_step}")
                    print("Batch Losses:", ", ".join([f"{n}={batch_losses[n].item():.4f}" for n in batch_losses]))
                    print(f"Adaptive weights: {assigned_weights}")
                    print("--------------------\n")
                    stop_epoch = True
                    if if_freeze_student:
                        print(">>> FREEZING STUDENT MODEL (all stop-thresholds satisfied) <<<")
                        self.student.freeze_student_model()
                    break

            epoch_avgs = {
                n: (epoch_losses[n] / max(1, epoch_batch_counts[n]))
                for n in epoch_losses
            }
            print(f"Epoch {epoch + 1} averages:", ", ".join([f"{n}={v:.4f}" for n, v in epoch_avgs.items()]))

            if stop_epoch:
                print(f"[Info] KD skipping triggered. Ending epoch {epoch + 1} early.")
                break

        visualize_student_performance(
            self.threshold_policy.student_performance,
            self.threshold_policy.config.relative_gain_stop_threshold,
            loss_names
        )

        return {
            "teacher": self.teacher,
            "student": self.student,
            "loss_history": loss_history,
            "final_loss_vector": epoch_avgs,
            "epochs_ran": epochs,
            "samples_per_epoch": stopped,
        }

    def _fuse_scores(self, student_score, teacher_score, threshold):
        return float((threshold * student_score +  teacher_score) / (1 + threshold))

    def generate_combined_feedback(
        self,
        prompt_text,
        answer,
        is_math=False,
        examples=None,
        epochs=1,
        iterations=20,
        threshold=0.65,
        KD=True,
        update_baseline=True
    ):
        result = compute_all_losses_vectorized(
            self.loss_config,
            prompt_text, answer,
            self.student, self.teacher,
            self.student_tokenizer, self.teacher_tokenizer,
            self.align_hidden_for_hidden_loss, self.projection_layer,
            device=self.device,
            is_math=is_math, examples=examples,
            model_dtype=self.model_dtype
        )

        student_feedback, teacher_feedback, student_score, teacher_score = unpack_feedback_and_scores(result)

        is_skipping, info = self.threshold_policy.should_skip_kd(result['loss_vector'], KD, update_baseline)
        adaptive_weights = info['adaptive_weights']
        if self.student.student_frozen:
            print("[Info] Student model already frozen. Skipping distillation.")
            return {
                "expert_feedback": teacher_feedback,
                "amateur_feedback": student_feedback,
                "combined_score": (threshold * student_score +  teacher_score) / (1 + threshold),
                "loss_vector": result["loss_vector"]
            }

        if is_skipping:
            print("[Info] Skipping KD for this sample based on threshold criteria.")
            return {
                "expert_feedback": teacher_feedback,
                "amateur_feedback": student_feedback,
                "combined_score": self._fuse_scores(student_score, teacher_score, threshold),
                "loss_vector": result["loss_vector"]
            }

        if not KD:
            return {
                "expert_feedback": teacher_feedback,
                "amateur_feedback": student_feedback,
                "combined_score": self._fuse_scores(student_score, teacher_score, threshold),
                "loss_vector": result["loss_vector"]
            }
        else:
            print("[Distill Trigger] Initiating knowledge distillation...")
            self.knowledge_distillation_with_SFT(weights=adaptive_weights, epochs=epochs, stopped=iterations, is_math=is_math, examples=examples)

            updated_result = compute_all_losses_vectorized(
                self.loss_config,
                prompt_text, answer,
                self.student, self.teacher,
                self.student_tokenizer, self.teacher_tokenizer,
                self.align_hidden_for_hidden_loss, self.projection_layer, device=self.device, is_math=is_math, examples=examples
            )

            student_feedback, teacher_feedback, student_score, teacher_score = unpack_feedback_and_scores(updated_result)

        combined_score = self._fuse_scores(student_score, teacher_score, threshold)

        return {
            "expert_feedback": teacher_feedback,
            "amateur_feedback": student_feedback,
            "combined_score": combined_score,
            "loss_vector": updated_result["loss_vector"]
        }
