import re
import json
import math
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from collections import Counter
import textstat
from sentence_transformers import SentenceTransformer, util

# Global SpaCy loaded lazily
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

# Global SentenceTransformer model loaded lazily
_emb_model = None
def get_emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _emb_model

class RefinedAnswerEvaluator:
    def __init__(self, tokenizer, model, use_bf16=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if self.device == "cuda" else torch.float32
        self.model = model

    def generate_text(self, prompt, max_new_tokens=50, temperature=0.3):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=0.5,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def is_refined_better(self, original_answer, refined_answer):
        prompt = (
            "You are a teacher grading answers to evaluate which sentence is better.\n"
            "Your goal is to provide a 'Yes' or 'No' answer.\n"
            "Criteria to evaluate the refined answer over the original answer:\n"
            "- Clarity: Is the refined sentence easier to understand?\n"
            "- Persuasiveness: Does the refined answer convince the reader more?\n"
            "- Engagement: Does the refined answer keep the reader interested?\n\n"
            "Original answer: The cat sat on the mat.\n"
            "Refined answer: The cat comfortably rested on the mat.\n"
            "Better? Yes\n\n"
            "Original answer: Water boils at 100 degrees Celsius.\n"
            "Refined answer: Water boils at 0 degrees Celsius.\n"
            "Better? No\n\n"
            f"Original answer: {original_answer}\n"
            f"Refined answer: {refined_answer}\n"
            "Better?"
        )
        output = self.generate_text(prompt, max_new_tokens=80)
        matches = re.findall(r"\bYes\b|\bNo\b", output, re.IGNORECASE)
        if matches:
            # Matches the target slot (usually 5th item if following structure) or defaults
            for match in reversed(matches):
                if match.lower() in ["yes", "no"]:
                    return match.lower() == "yes"
        return False

    def generate_engagement_score(self, text, max_new_tokens=10):
        prompt = (
            "You are a teacher grading answers to evaluate how engaging a sentence or paragraph is.\n"
            "On a scale of 1 to 10, use the following criteria:\n"
            "1: Extremely boring; reader would likely lose interest immediately.\n"
            "2: Very unengaging; dull and lacks clarity or appeal.\n"
            "3: Somewhat unengaging; few interesting elements.\n"
            "4: Slightly engaging; may hold attention briefly.\n"
            "5: Moderately engaging; neither particularly boring nor exciting.\n"
            "6: Fairly engaging; keeps the reader’s attention most of the time.\n"
            "7: Engaging; reader finds it interesting and clear.\n"
            "8: Very engaging; strong clarity, persuasiveness, and appeal.\n"
            "9: Extremely engaging; highly compelling and keeps reader fully interested.\n"
            "10: Exceptionally engaging; captivating, persuasive, and very clear, leaving a strong impact.\n\n"
            "Text: 'The cat sat quietly on the mat.'\n"
            "Score: 3\n\n"
            f"Text: {text}.\n"
            "Score:"
        )
        output = self.generate_text(prompt, max_new_tokens)
        score_matches = re.findall(r"Score\s*:\s*(\d+\.?\d*)", output)
        if score_matches:
            return float(score_matches[-1])
        return 0.0

def normalize_readability(score, text, informativeness, alpha=0.5):
    base = textstat.flesch_reading_ease(text)
    adjusted = (1 - alpha) * base + alpha * informativeness
    return adjusted

def engagement_score(text, evaluator):
    nlp = get_nlp()
    doc = nlp(text)
    verbs = sum(1 for token in doc if token.pos_ == "VERB")
    adjectives = sum(1 for token in doc if token.pos_ == "ADJ")
    length = len(doc)

    spacy_score = (verbs * 1.5 + adjectives * 1.0) / max(length, 1)
    llm_score_raw = evaluator.generate_engagement_score(text)
    llm_score = (llm_score_raw - 1) / 9.0

    combined_score = 0.5 * spacy_score + 0.5 * llm_score
    return combined_score

def specificity_score(text):
    nlp = get_nlp()
    doc = nlp(text)
    content_words = sum(1 for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"})
    return content_words / max(len(doc), 1)

def informativeness_score(text):
    nlp = get_nlp()
    tokens = [t.text.lower() for t in nlp(text) if not t.is_punct and not t.is_space]
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c/total for c in counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    return entropy / math.log(len(counts)+1, 2)

def compute_semantic_similarity(original_answer, refined_answer):
    emb_model = get_emb_model()
    emb_orig = emb_model.encode(original_answer, convert_to_tensor=True)
    emb_refined = emb_model.encode(refined_answer, convert_to_tensor=True)
    similarity = util.cos_sim(emb_orig, emb_refined).item()
    return similarity

def compute_perplexity(text, teacher_model, teacher_tokenizer):
    tokens = teacher_tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(teacher_model.device)
    with torch.no_grad():
        outputs = teacher_model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

def is_refined_more_fluent(original_answer, refined_answer, model, tokenizer):
    original_ppl = compute_perplexity(original_answer, model, tokenizer)
    refined_ppl = compute_perplexity(refined_answer, model, tokenizer)
    return refined_ppl < original_ppl

def is_consistent(premise, hypothesis, nli_model, nli_tokenizer, threshold=0.5):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(nli_model.device)
    with torch.no_grad():
        outputs = nli_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        entailment_prob = probs[2].item()
        is_consistent_flag = entailment_prob >= threshold
    return is_consistent_flag

def comparative_testing_results_log(pairs, evaluator, teacher_model, teacher_tokenizer, nli_model, nli_tokenizer):
    results = []
    for original, refined in pairs:
        engagement_orig = engagement_score(original, evaluator)
        engagement_refined = engagement_score(refined, evaluator)

        specificity_orig = specificity_score(original)
        specificity_refined = specificity_score(refined)

        informativeness_orig = informativeness_score(original)
        informativeness_refined = informativeness_score(refined)

        readability_orig = normalize_readability(
            textstat.flesch_reading_ease(original), original, informativeness_orig)

        readability_refined = normalize_readability(
            textstat.flesch_reading_ease(refined), refined, informativeness_refined)

        semantic_similarity = compute_semantic_similarity(original, refined)

        llm_pref = 1 if evaluator.is_refined_better(original, refined) else -1

        is_refined_easier_to_read = 1 if is_refined_more_fluent(original, refined, teacher_model, teacher_tokenizer) else 0
        is_refined_consistent = 1 if is_consistent(refined, original, nli_model, nli_tokenizer) else 0

        results.append({
            "original_readability": readability_orig,
            "refined_readability": readability_refined,
            "original_engagement": engagement_orig,
            "refined_engagement": engagement_refined,
            "original_specificity": specificity_orig,
            "refined_specificity": specificity_refined,
            "specificity_better": int(specificity_refined > specificity_orig),
            "original_informativeness": informativeness_orig,
            "refined_informativeness": informativeness_refined,
            "informativeness_better": int(informativeness_refined > informativeness_orig),
            "semantic_similarity": semantic_similarity,
            "llm_pref": llm_pref,
            "is_refined_fluent": is_refined_easier_to_read,
            "is_consistent": is_refined_consistent,
            "readability_better": int(readability_refined > readability_orig),
            "engagement_better": int(engagement_refined > engagement_orig),
            "fluent_better": is_refined_easier_to_read,
        })
    return results

def evaluate_pairs_with_frequentist(log_results, output_file="statistical_testing.json", alpha=0.05):
    votes = []
    decisions = []

    features = [
        "readability_better",
        "engagement_better",
        "specificity_better",
        "informativeness_better",
        "fluent_better"
    ]

    metric_data = []

    for r in log_results:
        metric_votes = [r.get(f, 0) for f in features]
        vote_sum = sum(metric_votes)
        metric_majority_vote = 1 if vote_sum > 0 else -1
        llm_pref = r.get("llm_pref", 0)

        if_match = int(llm_pref == metric_majority_vote)
        votes.append(if_match)
        metric_data.append(metric_votes + [llm_pref])

        decisions.append({
            "llm_pref": llm_pref,
            "metric_majority_vote": metric_majority_vote,
            "match": bool(if_match)
        })

    t_stat, p_value = ttest_1samp(votes, 0.5, alternative='greater')
    final_decision = bool(p_value < alpha)

    output_data = {
        "votes": votes,
        "decisions": decisions,
        "p_value": float(p_value),
        "final_decision": final_decision
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    df_for_corr = pd.DataFrame(metric_data, columns=features + ["llm_pref"])
    corr_matrix = df_for_corr.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()

    return final_decision

def _normalize_answer(s):
    s = s.strip().lower()
    s = re.sub(r'(?<=\d)[,\s](?=\d)', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[.,;:\s]+$', '', s)
    return s

_num_pat = re.compile(r'^[-+]?\d+(?:\.\d+)?%?$')

def _canon_number(s):
    t = re.sub(r'(?<=\d)[,\s](?=\d)', '', s.strip().lower())
    return t

def _is_correct(pred, gt):
    if pred is None or gt is None:
        return 0
    pred_n = _normalize_answer(pred)

    def _match(one, two):
        a = _normalize_answer(one)
        b = _normalize_answer(two)
        if _num_pat.match(a) and _num_pat.match(b):
            return _canon_number(a) == _canon_number(b)
        return a == b

    if isinstance(gt, (list, tuple, set)):
        return int(any(_match(pred, x) for x in gt))
    return int(_match(pred, gt))
