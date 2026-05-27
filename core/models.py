import re
import torch
import torch.nn as nn

def truncate_to_first_paragraph_expert(text):
    for part in text.strip().split('\n\n'):
        paragraph = part.strip().split('\n')[0].strip()
        if paragraph:
            return paragraph
    return text.strip()

class ExpertFeedbackModel:
    def __init__(self, tokenizer, model, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded {model_name} on {self.device}")

    def generate_text(self, prompt, max_new_tokens=300):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_answer(self, question, is_math=False, task_description=None, examples=None, max_new_tokens=500, max_words=300):
        question = question.strip()
        task_description = task_description.strip() if task_description else None

        if is_math:
            task_text = f"Task Description: {task_description}\n" if task_description else ""
            prompt = (
                "You are a specialized math language model tasked with generating accurate, "
                "well-reasoned, step-by-step solutions to specific math problems.\n"
                "Your goal is to provide clear, precise, and fully self-contained explanations "
                "suitable for anyone seeking a solid understanding of the topic, regardless of prior background knowledge.\n"
                "When responding:\n"
                "  1) Identify the mathematical concept or principle involved.\n"
                "  2) Solve the problem step by step.\n"
                "  3) Present the solution in a logically structured and concise manner.\n"
                "Keep the tone professional, formal, concise and focused. Avoid filler words, emojis, or informal phrasing.\n"
                "Write your answer in a single paragraph.\n"
                f"Limit your response to approximately {max_words} words.\n\n"
                "Think carefully before responding. Format your output as follows:\n\n"
                "Answer: <one-paragraph response>\n\n"
                f"{task_text}"
                f"Question: {question}\n"
                "Answer:"
            )
        else:
            prompt = (
                "You are an expert language model tasked with providing precise, well-reasoned, and informative responses.\n"
                "Your goal is to deliver a clear, accessible, and accurate response in a formal yet approachable tone for a general audience with no assumed background knowledge.\n"
                "The user is an informed individual seeking a reliable, well-reasoned, and self-contained explanation.\n"
                "Your response will be used to inform readers seeking a foundational understanding of the topic.\n"
                "When responding to the request: 1. Identify the core message. 2. Explain it in a way that is logically structured, accurate, and easy to understand.\n"
                "Keep the tone professional and concise. Avoid jargon, filler words, and emojis.\n"
                f"Limit your response to approximately {max_words} words. If a full explanation would be too long, summarize it while ensuring it remains self-contained and informative.\n"
                "Write in a single paragraph with no lists, bullet points, or follow-up suggestions.\n"
                "Think step by step to ensure your response improves clarity and aligns with the target audience.\n\n"
                f"Task Description: {task_description}\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

        response = self.generate_text(prompt, max_new_tokens)
        blocks = response.split("Answer:")
        if not is_math:
            answer = blocks[1].strip() if len(blocks) >= 2 else ""
        else:
            answer = blocks[2].strip() if len(blocks) >= 3 else ""

        return truncate_to_first_paragraph_expert(answer)

    def generate_feedback(self, question, answer, is_math=False, examples=None):
        if is_math and examples:
            one_shot_example = examples[0]
            one_shot_question = one_shot_example.get('question', '')
            one_shot_answer = one_shot_example.get('answer', '')
            one_shot_score = one_shot_example.get('score', '')
            one_shot_feedback = one_shot_example.get('expert_feedback', '')
            one_shot_text = f"Question: {one_shot_question}\nAnswer: {one_shot_answer}\nScore: {one_shot_score}\nFeedback: {one_shot_feedback}\n"
        else:
            one_shot_text = (
                "Question: How does the author use symbolism in the story to convey the protagonist's emotional journey?\n"
                "Answer: The author uses the image of the broken mirror to represent how the protagonist feels fractured internally. "
                "Each mention of the mirror reflects a different stage of the character’s turmoil. "
                "Initially, the mirror is intact and clear, but as the story progresses, it becomes cracked and dusty, symbolizing the descent into depression and self-doubt. "
                "The mirror also highlights how the character perceives themselves — distorted and unclear due to emotional pain.\n\n"
                "Score: 0.85\n"
                "Feedback: This answer demonstrates a perceptive and coherent interpretation of the mirror as a symbol of the protagonist’s emotional state. "
                "The symbolic progression is clearly articulated, and the discussion of distortion is particularly effective. "
                "To improve, the analysis should incorporate specific textual evidence. "
                "Overall, this is a strong and emotionally insightful response.\n"
            )

        prompt = (
            "You are an expert feedback evaluator. Your task is to assess the quality and relevance of the answer provided below.\n"
            "The answer may involve subjective analysis (e.g., literature, interpretation) or objective reasoning (e.g., math, science).\n\n"
            "Follow these steps:\n"
            "1. Carefully analyze the answer. Internally identify its strengths, weaknesses, and areas for improvement.\n"
            "2. Write a clear, specific, and constructive paragraph of feedback that would help the user improve their response.\n"
            "3. Output your final **Score** and **Feedback** in the format shown below.\n\n"
            "Use a professional, concise, and objective tone. Do not include personal encouragements or vague generalities.\n"
            "Use the following scoring scale:\n"
            "  •  1.0 — Excellent: insightful, emotionally engaging, and well-structured\n"
            "  •  0.7–0.9 — Good: mostly clear and thoughtful with minor issues\n"
            "  •  0.4–0.6 — Fair: has substance but lacks clarity or cohesion\n"
            "  •  0.1–0.3 — Weak: vague, underdeveloped, or emotionally flat\n"
            "  •  0.0 — Not helpful: irrelevant or incoherent\n"
            "  • -0.1 to -0.4 — Somewhat misleading or confusing\n"
            "  • -0.5 to -0.9 — Mostly incorrect, misrepresents the source\n"
            "  • -1.0 — Harmful or dangerously misleading\n\n"
            "Think carefully before responding. Format your output as follows:\n\n"
            "Score: <number>\n"
            "Feedback: <one-paragraph response>\n\n"
            f"{one_shot_text}"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Score:"
        )

        response = self.generate_text(prompt, max_new_tokens=200).strip()
        score_blocks = response.split("Score:")
        score_text = score_blocks[3].strip().split()[0] if len(score_blocks) >= 4 else (score_blocks[1].strip().split()[0] if len(score_blocks) >= 2 else None)

        try:
            score = float(score_text)
        except (TypeError, ValueError):
            score = -1.0

        feedback_blocks = response.split("Feedback:")
        feedback_blocks = [f for f in feedback_blocks if '<' not in f and '>' not in f]
        feedback_text_expert = truncate_to_first_paragraph_expert(feedback_blocks[1].strip()) if len(feedback_blocks) >= 2 else ""

        return feedback_text_expert, score

    def generate_unified_feedback(self, expert_feedback, amateur_feedback, max_new_tokens=200):
        if amateur_feedback == "Failed to generate meaningful feedback":
            return expert_feedback.strip()

        prompt = (
            f"You are a feedback summarizer tasked with condensing two feedback comments on an answer.\n"
            f"Your goal is to generate a clear and concise summary by prioritizing the expert feedback while incorporating any useful or original points from the amateur feedback.\n"
            f"Write in a formal yet approachable tone, suitable for a general audience with no assumed background knowledge.\n"
            f"The user is an informed individual seeking a balanced summary to improve their response.\n"
            f"Your summary should provide informative and actionable feedback.\n"
            f"In cases of disagreement, favor the expert’s insights and ensure the final version is coherent, constructive, and no more than 70 words.\n\n"
            f"You are given two feedback comments on an answer:\n"
            f"Your task is to write a unified feedback summary (max 70 words) that:\n"
            f"- Prioritizes insights from the expert feedback\n"
            f"- Incorporates any unique and constructive points from the amateur feedback\n"
            f"- Resolves disagreements by favoring the expert\n"
            f"- Uses contrastive reasoning to emphasize the strengths of the expert feedback\n\n"
            f"Think carefully before responding. Format your output as follows:\n\n"
            f"Final Merged Feedback: <one-paragraph response>\n\n"
            f"Expert Feedback: The argument is clear and well-structured, but the conclusion could be stronger with more supporting data. Consider adding one more example to illustrate the key point.\n"
            f"Amateur Feedback: I liked how the answer flows, but maybe shorten some sentences to make it punchier.\n"
            f"Final Merged Feedback: The answer is clear, with strong logical flow. To enhance impact, strengthen the conclusion by adding more supporting data and one illustrative example. Additionally, minor sentence shortening can improve readability without losing clarity.\n\n"
            f"Expert Feedback: {expert_feedback.strip()}\n"
            f"Amateur Feedback: {amateur_feedback.strip()}\n"
            f"Final Merged Feedback:"
        )

        response = self.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
        feedback_blocks = response.split("Final Merged Feedback:")
        feedback_blocks = [f.strip() for f in feedback_blocks if '<' not in f and '>' not in f and f.strip()]

        if len(feedback_blocks) >= 3:
            return truncate_to_first_paragraph_expert(feedback_blocks[2].strip())
        return truncate_to_first_paragraph_expert(response.strip())

    def apply_feedback(self, question, answer, feedback, max_new_tokens=500):
        prompt = (
            "You are an expert language model tasked with producing precise, well-reasoned, and informative responses.\n"
            "Your goal is to revise the original answer using the provided feedback to create a response that is clear, accurate, and accessible.\n"
            "Write in a formal yet approachable tone, suitable for a general audience with no assumed background knowledge.\n"
            "The user is an informed individual seeking a reliable, logically structured, and self-contained explanation.\n"
            "Your revised response will help readers develop a foundational understanding of the topic.\n"
            "When crafting the response: 1. Identify the core message. 2. Present it in a logically structured, accurate, and easy-to-understand manner.\n"
            "Maintain a professional and concise tone. Avoid jargon, filler words, and emojis.\n"
            "Write your answer in a single paragraph.\n"
            "Limit your response to approximately 200 words.\n\n"
            "Think carefully before responding. Format your output as follows:\n\n"
            "Revised Answer: <one-paragraph response>\n\n"
            "Revise the following answer using the feedback provided.\n"
            f"Question:{question}\n"
            f"Original Answer:{answer}\n"
            f"Feedback:{feedback}\n"
            "Revised Answer:"
        )

        response = self.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
        answer_blocks = response.split("Revised Answer:")
        answer_blocks = [a.strip() for a in answer_blocks if '<' not in a and '>' not in a and a.strip()]

        final_answer = answer_blocks[1].strip() if len(answer_blocks) >= 2 else response.strip()
        return truncate_to_first_paragraph_expert(final_answer)

    def generate_self_critique(self, question, answer, max_new_tokens=150):
        prompt = (
            "You are the author of the provided answer. Your task is to critically evaluate your own response in a concise, paragraph-style self-critique.\n"
            "You should analyze your answer for clarity, accuracy, completeness, and relevance to the question.\n"
            "You should identify the strengths of your response and explain why certain parts effectively address the question.\n"
            "You should also highlight weaknesses or gaps, and suggest how you could have improved the answer.\n"
            "You should not rewrite the answer; focus on evaluating it objectively from your own perspective.\n"
            "You should write in a formal yet approachable tone, suitable for a general audience.\n"
            "You should limit your critique to approximately 70 words, focusing on the most important points.\n\n"
            "Format the output as follows:\n"
            "Self-Critique: <one-paragraph response>\n\n"
            "Question: What are the main factors contributing to climate change?\n"
            "Answer: Climate change is caused primarily by human activities, such as burning fossil fuels and deforestation, which increase greenhouse gas concentrations in the atmosphere. Natural factors like volcanic eruptions and solar variability also play minor roles.\n"
            "Self-Critique: I correctly identified human activities as the primary driver of climate change, which is accurate and relevant. I acknowledged natural factors, offering balance. However, I could improve by naming specific greenhouse gases and their impacts, which would make the response more complete. Overall, my answer is accurate but could be more detailed and precise.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Self-Critique:"
        )

        response = self.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
        critique_blocks = response.split("Self-Critique:")
        critique_blocks = [c.strip() for c in critique_blocks if '<' not in c and '>' not in c and c.strip()]

        return truncate_to_first_paragraph_expert(critique_blocks[2].strip()) if len(critique_blocks) >= 3 else ""

    def apply_self_critique(self, question, answer, self_critique, max_new_tokens=500):
        prompt = (
            "You are an expert language model tasked with producing clear, accurate, and well-reasoned answers.\n"
            "Your goal is to revise the following answer based on the provided self-critique, "
            "ensuring the revision is logically sound, clear, and self-contained.\n\n"
            "Write in a formal yet approachable tone, suitable for a general audience with no assumed background knowledge.\n"
            "The user is an informed individual seeking a reliable, logically structured, and self-contained explanation.\n\n"
            "When crafting the response:\n"
            "1. Incorporate all insights from the self-critique.\n"
            "2. Improve clarity and logical flow.\n"
            "3. Correct any errors or ambiguities in the original answer.\n"
            "4. Keep the answer concise but thorough.\n\n"
            "Limit your response to approximately 200 words.\n\n"
            "Think carefully before responding. Format your output as follows:\n\n"
            "Final Answer: <one-paragraph response>\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Self-Critique: {self_critique}\n"
            "Final Answer:"
        )

        response = self.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
        answer_blocks = response.split("Final Answer:")
        answer_blocks = [a.strip() for a in answer_blocks if '<' not in a and '>' not in a and a.strip()]

        return truncate_to_first_paragraph_expert(answer_blocks[1].strip()) if len(answer_blocks) >= 2 else ""

    def improve_answer_with_feedback_and_critique(self, question, answer, epochs, iterations, target_score=1.0, mode="train", is_math=False, examples=None):
        KD = True if mode == "train" else False
        original_feedback_dict = self.network.generate_combined_feedback(question, answer, is_math, examples, epochs, iterations, 0.65, KD)

        expert_feedback = original_feedback_dict.get("expert_feedback", "")
        amateur_feedback = original_feedback_dict.get("amateur_feedback", "")
        initial_score = float(original_feedback_dict.get("combined_score", -1.0))

        if initial_score >= target_score:
            return answer.strip(), expert_feedback, amateur_feedback, ""

        condensed_feedback = self.generate_unified_feedback(expert_feedback, amateur_feedback)
        feedback_paragraph = condensed_feedback.strip().split('\n\n')[0].strip()

        improved = self.apply_feedback(question, answer, feedback_paragraph)
        critique = self.generate_self_critique(question, improved)
        refined = self.apply_self_critique(question, improved, critique)

        return refined.strip(), expert_feedback, amateur_feedback, feedback_paragraph


class AmateurFeedbackModel:
    def __init__(self, tokenizer, model, model_name, use_bf16=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = model
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_dim = self.model.config.hidden_size
        self.model_dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.score_head = nn.Linear(hidden_dim, 1, dtype=self.model_dtype).to(self.device)
        print(f"Loaded {model_name} on {self.device} with dtype {self.model_dtype}")
        self.student_frozen = False

    def generate_text(self, prompt, max_new_tokens=300):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Added Phase 1 Methods for Autonomous Inference
    def generate_answer(self, question, is_math=False, task_description=None, examples=None, max_new_tokens=500, max_words=300):
        question = question.strip()
        task_description = task_description.strip() if task_description else None

        if is_math:
            task_text = f"Task Description: {task_description}\n" if task_description else ""
            prompt = (
                "You are a specialized math language model tasked with generating accurate, "
                "well-reasoned, step-by-step solutions to specific math problems.\n"
                "Your goal is to provide clear, precise, and fully self-contained explanations "
                "suitable for anyone seeking a solid understanding of the topic, regardless of prior background knowledge.\n"
                "When responding:\n"
                "  1) Identify the mathematical concept or principle involved.\n"
                "  2) Solve the problem step by step.\n"
                "  3) Present the solution in a logically structured and concise manner.\n"
                "Keep the tone professional, formal, concise and focused. Avoid filler words, emojis, or informal phrasing.\n"
                "Write your answer in a single paragraph.\n"
                f"Limit your response to approximately {max_words} words.\n\n"
                "Think carefully before responding. Format your output as follows:\n\n"
                "Answer: <one-paragraph response>\n\n"
                f"{task_text}"
                f"Question: {question}\n"
                "Answer:"
            )
        else:
            prompt = (
                "You are an expert language model tasked with providing precise, well-reasoned, and informative responses.\n"
                "Your goal is to deliver a clear, accessible, and accurate response in a formal yet approachable tone for a general audience with no assumed background knowledge.\n"
                "The user is an informed individual seeking a reliable, well-reasoned, and self-contained explanation.\n"
                "Your response will be used to inform readers seeking a foundational understanding of the topic.\n"
                "When responding to the request: 1. Identify the core message. 2. Explain it in a way that is logically structured, accurate, and easy to understand.\n"
                "Keep the tone professional and concise. Avoid jargon, filler words, and emojis.\n"
                f"Limit your response to approximately {max_words} words. If a full explanation would be too long, summarize it while ensuring it remains self-contained and informative.\n"
                "Write in a single paragraph with no lists, bullet points, or follow-up suggestions.\n"
                "Think step by step to ensure your response improves clarity and aligns with the target audience.\n\n"
                f"Task Description: {task_description}\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

        response = self.generate_text(prompt, max_new_tokens)
        blocks = response.split("Answer:")
        if not is_math:
            answer = blocks[1].strip() if len(blocks) >= 2 else ""
        else:
            answer = blocks[2].strip() if len(blocks) >= 3 else ""

        return truncate_to_first_paragraph_expert(answer)

    def apply_feedback(self, question, answer, feedback, max_new_tokens=500):
        prompt = (
            "You are an expert language model tasked with producing precise, well-reasoned, and informative responses.\n"
            "Your goal is to revise the original answer using the provided feedback to create a response that is clear, accurate, and accessible.\n"
            "Write in a formal yet approachable tone, suitable for a general audience with no assumed background knowledge.\n"
            "The user is an informed individual seeking a reliable, logically structured, and self-contained explanation.\n"
            "Your revised response will help readers develop a foundational understanding of the topic.\n"
            "When crafting the response: 1. Identify the core message. 2. Present it in a logically structured, accurate, and easy-to-understand manner.\n"
            "Maintain a professional and concise tone. Avoid jargon, filler words, and emojis.\n"
            "Write your answer in a single paragraph.\n"
            "Limit your response to approximately 200 words.\n\n"
            "Think carefully before responding. Format your output as follows:\n\n"
            "Revised Answer: <one-paragraph response>\n\n"
            "Revise the following answer using the feedback provided.\n"
            f"Question:{question}\n"
            f"Original Answer:{answer}\n"
            f"Feedback:{feedback}\n"
            "Revised Answer:"
        )

        response = self.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
        answer_blocks = response.split("Revised Answer:")
        answer_blocks = [a.strip() for a in answer_blocks if '<' not in a and '>' not in a and a.strip()]

        final_answer = answer_blocks[1].strip() if len(answer_blocks) >= 2 else response.strip()
        return truncate_to_first_paragraph_expert(final_answer)

    def generate_self_critique(self, question, answer, max_new_tokens=150):
        prompt = (
            "You are the author of the provided answer. Your task is to critically evaluate your own response in a concise, paragraph-style self-critique.\n"
            "You should analyze your answer for clarity, accuracy, completeness, and relevance to the question.\n"
            "You should identify the strengths of your response and explain why certain parts effectively address the question.\n"
            "You should also highlight weaknesses or gaps, and suggest how you could have improved the answer.\n"
            "You should not rewrite the answer; focus on evaluating it objectively from your own perspective.\n"
            "You should write in a formal yet approachable tone, suitable for a general audience.\n"
            "You should limit your critique to approximately 70 words, focusing on the most important points.\n\n"
            "Format the output as follows:\n"
            "Self-Critique: <one-paragraph response>\n\n"
            "Question: What are the main factors contributing to climate change?\n"
            "Answer: Climate change is caused primarily by human activities, such as burning fossil fuels and deforestation, which increase greenhouse gas concentrations in the atmosphere. Natural factors like volcanic eruptions and solar variability also play minor roles.\n"
            "Self-Critique: I correctly identified human activities as the primary driver of climate change, which is accurate and relevant. I acknowledged natural factors, offering balance. However, I could improve by naming specific greenhouse gases and their impacts, which would make the response more complete. Overall, my answer is accurate but could be more detailed and precise.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Self-Critique:"
        )

        response = self.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
        critique_blocks = response.split("Self-Critique:")
        critique_blocks = [c.strip() for c in critique_blocks if '<' not in c and '>' not in c and c.strip()]

        return truncate_to_first_paragraph_expert(critique_blocks[2].strip()) if len(critique_blocks) >= 3 else ""

    def apply_self_critique(self, question, answer, self_critique, max_new_tokens=500):
        prompt = (
            "You are an expert language model tasked with producing clear, accurate, and well-reasoned answers.\n"
            "Your goal is to revise the following answer based on the provided self-critique, "
            "ensuring the revision is logically sound, clear, and self-contained.\n\n"
            "Write in a formal yet approachable tone, suitable for a general audience with no assumed background knowledge.\n"
            "The user is an informed individual seeking a reliable, logically structured, and self-contained explanation.\n\n"
            "When crafting the response:\n"
            "1. Incorporate all insights from the self-critique.\n"
            "2. Improve clarity and logical flow.\n"
            "3. Correct any errors or ambiguities in the original answer.\n"
            "4. Keep the answer concise but thorough.\n\n"
            "Limit your response to approximately 200 words.\n\n"
            "Think carefully before responding. Format your output as follows:\n\n"
            "Final Answer: <one-paragraph response>\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Self-Critique: {self_critique}\n"
            "Final Answer:"
        )

        response = self.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
        answer_blocks = response.split("Final Answer:")
        answer_blocks = [a.strip() for a in answer_blocks if '<' not in a and '>' not in a and a.strip()]

        return truncate_to_first_paragraph_expert(answer_blocks[1].strip()) if len(answer_blocks) >= 2 else ""

    def generate_feedback(self, question, answer, is_math=False, examples=None):
        if is_math and examples:
            one_shot_example = examples[0]
            one_shot_question = one_shot_example.get('question', '')
            one_shot_answer = one_shot_example.get('answer', '')
            one_shot_feedback = one_shot_example.get('expert_feedback', '')
            one_shot_text = f"Question: {one_shot_question}\nAnswer: {one_shot_answer}\nFeedback: {one_shot_feedback}\n"
        else:
            one_shot_text = (
                "Question: How does the author use symbolism in the story to convey the protagonist's emotional journey?\n"
                "Answer: The author uses the image of the broken mirror to represent how the protagonist feels fractured internally. "
                "Each mention of the mirror reflects a different stage of the character’s turmoil. "
                "Initially, the mirror is intact and clear, but as the story progresses, it becomes cracked and dusty, symbolizing the descent into depression and self-doubt. "
                "The mirror also highlights how the character perceives themselves — distorted and unclear due to emotional pain.\n\n"
                "Feedback: This answer demonstrates a perceptive and coherent interpretation of the mirror as a symbol of the protagonist’s emotional state. "
                "The symbolic progression is clearly articulated, and the discussion of distortion is particularly effective. "
                "To improve, the analysis should incorporate specific textual evidence. "
                "Overall, this is a strong and emotionally insightful response.\n\n"
            )
        prompt = (
            "You are an expert feedback evaluator. Your task is to assess the quality and relevance of the provided answer.\n"
            "The answer may involve subjective analysis (e.g., literature, interpretation) or objective reasoning (e.g., math, science).\n\n"
            "Instructions:\n"
            "1. Identify the answer’s strengths, weaknesses, and areas for improvement.\n"
            "2. Write a clear, specific, and constructive paragraph of feedback.\n"
            "3. Provide the answer in plain text only, with no special characters or markdown.\n"
            "4. Use a professional, concise, and objective tone. Avoid encouragements or vague comments.\n"
            "5. Keep the response around 70 words.\n\n"
            "Think carefully before responding. Format your output as follows:\n\n"
            "Feedback: <one-paragraph response>\n\n"
            f"{one_shot_text}"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Feedback:"
        )

        response = self.generate_text(prompt, max_new_tokens=200).strip()
        feedback_blocks = response.split("Feedback:")
        feedback_blocks = [f for f in feedback_blocks if '<' not in f and '>' not in f]
        feedback_text = truncate_to_first_paragraph_expert(feedback_blocks[2].strip()) if len(feedback_blocks) >= 3 else ""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        last_hidden_state = last_hidden_state.to(self.score_head.weight.dtype)
        raw_score = self.score_head(last_hidden_state)
        score_logits = torch.tanh(raw_score)
        score_value = score_logits.detach().item()

        return feedback_text, score_value, score_logits

    def prepare_inputs_and_labels(self, prompt, answer, expert_feedback, is_math=False, examples=None):
        if is_math and examples:
            one_shot_example = examples[0]
            one_shot_question = one_shot_example.get('question', '')
            one_shot_answer = one_shot_example.get('answer', '')
            one_shot_feedback = one_shot_example.get('expert_feedback', '')
            one_shot_text = f"Question: {one_shot_question}\nAnswer: {one_shot_answer}\nFeedback: {one_shot_feedback}\n"
        else:
            one_shot_text = (
                "Question: How does the author use symbolism in the story to convey the protagonist's emotional journey?\n"
                "Answer: The author uses the image of the broken mirror to represent how the protagonist feels fractured internally. "
                "Each mention of the mirror reflects a different stage of the character’s turmoil. "
                "Initially, the mirror is intact and clear, but as the story progresses, it becomes cracked and dusty, symbolizing the descent into depression and self-doubt. "
                "The mirror also highlights how the character perceives themselves — distorted and unclear due to emotional pain.\n\n"
                "Feedback: This answer demonstrates a perceptive and coherent interpretation of the mirror as a symbol of the protagonist’s emotional state. "
                "The symbolic progression is clearly articulated, and the discussion of distortion is particularly effective. "
                "To improve, the analysis should incorporate specific textual evidence. "
                "Overall, this is a strong and emotionally insightful response.\n\n"
            )

        input_text = (
            "You are an expert feedback evaluator. Your task is to assess the quality and relevance of the provided answer.\n"
            "The answer may involve subjective analysis (e.g., literature, interpretation) or objective reasoning (e.g., math, science).\n\n"
            "Instructions:\n"
            "1. Identify the answer’s strengths, weaknesses, and areas for improvement.\n"
            "2. Write a clear, specific, and constructive paragraph of feedback.\n"
            "3. Use this format: Feedback: <one-paragraph response>.\n"
            "4. Provide the answer in plain text only, with no special characters or markdown.\n"
            "5. Use a professional, concise, and objective tone. Avoid encouragements or vague comments.\n"
            "6. Keep the response around 70 words.\n\n"
            f"{one_shot_text}"
            f"Question: {prompt}\n"
            f"Answer: {answer}\n\n"
            "Feedback:"
        )

        tokenized_input = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = tokenized_input.input_ids[0].to(self.device)

        tokenized_feedback = self.tokenizer(expert_feedback, return_tensors="pt", truncation=True, max_length=1024)
        feedback_ids = tokenized_feedback.input_ids[0].to(self.device)

        input_ids = torch.cat([
            input_ids,
            torch.full((len(feedback_ids),), self.tokenizer.pad_token_id, device=self.device)
        ])

        labels = torch.full_like(input_ids, -100)
        labels[-len(feedback_ids):] = feedback_ids

        attention_mask = torch.cat([
            tokenized_input.attention_mask[0].to(self.device),
            torch.ones(len(feedback_ids), device=self.device)
        ])

        return (
            input_ids.unsqueeze(0).to(self.device),
            labels.unsqueeze(0).to(self.device),
            attention_mask.unsqueeze(0).to(self.device)
        )

    def freeze_student_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.student_frozen = True


class ParsingModel:
    def __init__(self, tokenizer, model, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = model
        print(f"Loaded {model_name} on {self.device}")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_prompt(self, answer, is_math=False):
        if is_math:
            prompt = (
                "You are a mathematical answer extractor tasked with extracting the final numerical answer "
                "from a complicated formatted answer. Extract only the numerical value, which can be an integer, "
                "decimal, fraction (in a/b form), or percentage. Keep fractions in a/b format and keep % if present. "
                "Do not include any words, units, or additional context. Think carefully before responding. Format your output as follows:\n\n"
                "Extracted final numerical answer: <numerical value>\n\n"
                "Answer: The total cost is $120, plus tax, so \\boxed{120} is the amount before tax.\n"
                "Extracted final numerical answer: 120\n\n"
                f"Answer: {answer}\n"
                "Extracted final numerical answer:"
            )
        else:
            prompt = (
                "You are an answer extractor. From the given Answer text, copy the SHORTEST CONTIGUOUS SPAN "
                "that directly answers the question. Do NOT invent text. Output ONLY that span.\n\n"
                "Formatting:\n"
                "- Print exactly one line: 'Extracted final answer: <span>'\n"
                "- No extra words, no explanations, no quotes.\n\n"
                "Rules for difficult cases:\n"
                "1) DATES: If a month name (or abbreviation) appears anywhere in the Answer with a matching day, "
                "   prefer the full date phrase (e.g., 'September 23rd') over a bare day ('23'). Keep ordinals.\n"
                "2) NAMED ENTITIES WITH DETERMINERS: If the Answer’s concluding statement uses a determiner with "
                "   a proper noun (e.g., 'the Nile', 'the Netherlands'), KEEP it. For people (e.g., 'Abraham Lincoln'), "
                "   just the name.\n"
                "3) NUMBERS: If the final answer is a number, return just the number.\n"
                "4) COPY EXACTLY from the provided Answer text; do not normalize, translate, or change casing.\n\n"
                "Answer: The passage states that the capital of France is Paris, after discussing other cities.\n"
                "Extracted final answer: Paris\n\n"
                f"Answer: {answer}\n"
                "Extracted final answer:"
            )
        return prompt

    def generate_answer(self, answer, is_math=False, max_new_tokens=50):
        full_prompt = self.generate_prompt(answer, is_math)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        extracted = generated_text[len(full_prompt):].strip()
        return truncate_to_first_paragraph_expert(extracted)
