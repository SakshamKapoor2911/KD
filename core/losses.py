import torch
import torch.nn.functional as F

def compute_lm_loss(teacher_feedback, student, prompt, answer, device, model_dtype, is_math=False, examples=None):
    lm_input_ids, lm_labels, lm_attention_mask = student.prepare_inputs_and_labels(
                            prompt, answer, teacher_feedback, is_math, examples)

    lm_input_ids = lm_input_ids.to(device)
    lm_labels = lm_labels.to(device)
    lm_attention_mask = lm_attention_mask.to(device)

    lm_student_out = student.model(
        input_ids=lm_input_ids,
        attention_mask=lm_attention_mask,
        labels=lm_labels,
        output_hidden_states=True,
        return_dict=True
    )

    lm_loss = lm_student_out.loss.to(model_dtype)
    return lm_loss


def compute_hidden_loss(ref_text, student_tokenizer, teacher_tokenizer, student, teacher, align_hidden, device, model_dtype):
    student_inputs = student_tokenizer(ref_text, return_tensors="pt", truncation=True, padding=True).to(device)
    teacher_inputs = teacher_tokenizer(ref_text, return_tensors="pt", truncation=True, padding=True).to(device)

    student_out = student.model(**student_inputs, output_hidden_states=True, return_dict=True)
    with torch.no_grad():
        teacher_out = teacher.model(**teacher_inputs, output_hidden_states=True, return_dict=True)

    student_hidden, teacher_hidden = student_out.hidden_states, teacher_out.hidden_states
    num_student_layers, num_teacher_layers = len(student_hidden), len(teacher_hidden)
    layer_ids = torch.linspace(0, num_teacher_layers-1, num_student_layers).long()

    hidden_loss = 0.0

    for i, t_idx in enumerate(layer_ids):
        s_mean = student_hidden[i].to(model_dtype).mean(dim=1)
        t_mean = teacher_hidden[t_idx].to(model_dtype).mean(dim=1)

        s_norm = s_mean / s_mean.norm(dim=1, keepdim=True)
        t_norm = t_mean / t_mean.norm(dim=1, keepdim=True)

        s_aligned = align_hidden(s_norm)

        hidden_loss += F.open_loss(s_aligned, t_norm, reduction='sum') if hasattr(F, 'open_loss') else F.mse_loss(s_aligned, t_norm, reduction='sum')

    hidden_loss = hidden_loss / num_student_layers
    return hidden_loss

def compute_scoring_loss(score_pred, teacher_score, model_dtype, device):
    teacher_score_tensor = torch.tensor(
        [teacher_score] * score_pred.size(0),
        dtype=model_dtype,
        device=device
    )
    scoring_loss = F.mse_loss(score_pred.squeeze(-1), teacher_score_tensor)
    return scoring_loss

def z_score(x, tau=1.0, eps=1e-8):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + eps
    return (x - mean) / std / tau

def compute_logit_standardization(student, teacher, student_tokenizer, teacher_tokenizer, projection_layer, ref_text, tau=4.0):
    student_inputs = student_tokenizer(ref_text, return_tensors="pt", truncation=True).to(student.device)
    student_outputs = student.model(**student_inputs)
    student_logits = student_outputs.logits.to(torch.float32)
    student_logits_projected = projection_layer(student_logits)

    with torch.no_grad():
        teacher_inputs = teacher_tokenizer(ref_text, return_tensors="pt", truncation=True).to(student.device)
        teacher_outputs = teacher.model(**teacher_inputs)
        teacher_logits = teacher_outputs.logits.to(torch.float32)

    min_len = min(student_logits_projected.size(1), teacher_logits.size(1))
    student_logits_projected = student_logits_projected[:, :min_len, :]
    teacher_logits = teacher_logits[:, :min_len, :]

    teacher_logits_std = z_score(teacher_logits, tau)
    student_logits_std = z_score(student_logits_projected, tau)

    teacher_distribution = F.softmax(teacher_logits_std, dim=-1)
    student_distribution = F.log_softmax(student_logits_std, dim=-1)

    logits_loss = F.kl_div(student_distribution, teacher_distribution, reduction='batchmean') * (tau ** 2)
    return logits_loss

def pick_proportional_layers(student_total, teacher_total, n_layers=3):
    student_layers = [
        round(i * student_total / n_layers) for i in range(1, n_layers + 1)
    ]
    teacher_layers = [
        round(l * teacher_total / student_total) for l in student_layers
    ]
    return student_layers, teacher_layers

def compute_all_losses_vectorized(
    loss_config,
    prompt,
    answer,
    student,
    teacher,
    student_tokenizer,
    teacher_tokenizer,
    align_hidden_for_hidden_loss,
    projection_layer,
    device=None,
    is_math=False,
    examples=None,
    model_dtype=torch.float32
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_feedback, student_score_value, student_score_logit = student.generate_feedback(
        prompt, answer, is_math, examples
    )
    with torch.no_grad():
        teacher_feedback, teacher_score_value = teacher.generate_feedback(
            prompt, answer, is_math=is_math, examples=examples
        )

    active_losses, _ = loss_config.get_active_losses()
    losses_by_name = {}

    if "lm_loss" in active_losses:
        lm_loss = compute_lm_loss(
            teacher_feedback, student, prompt, answer, device, model_dtype, is_math, examples
        )
        losses_by_name["lm_loss"] = lm_loss

    if "hidden_loss" in active_losses:
        hidden_loss = compute_hidden_loss(
            teacher_feedback, student_tokenizer, teacher_tokenizer,
            student, teacher, align_hidden_for_hidden_loss, device, model_dtype
        )
        losses_by_name["hidden_loss"] = hidden_loss

    if "scoring_loss" in active_losses:
        teacher_score_t = torch.as_tensor(teacher_score_value, device=device, dtype=model_dtype)
        scoring_loss = compute_scoring_loss(
            student_score_logit, teacher_score_t, model_dtype, device
        )
        losses_by_name["scoring_loss"] = scoring_loss

    if "logit_loss" in active_losses:
        logit_loss = compute_logit_standardization(
            student, teacher, student_tokenizer, teacher_tokenizer,
            projection_layer, teacher_feedback, tau=4.0
        )
        losses_by_name["logit_loss"] = logit_loss

    student_loss_vector = torch.stack(
        [losses_by_name[name].to(model_dtype) for name in active_losses]
    ).to(device)

    return {
        "loss_vector": student_loss_vector,
        "student_feedback": student_feedback,
        "teacher_feedback": teacher_feedback,
        "student_score": student_score_value,
        "teacher_score": teacher_score_value
    }

def unpack_feedback_and_scores(result):
    return (
        result["student_feedback"],
        result["teacher_feedback"],
        result["student_score"],
        result["teacher_score"]
    )
