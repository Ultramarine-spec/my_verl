import re
import requests
from openai import OpenAI
from tqdm import tqdm

openai_api_key = "EMPTY"
openai_api_base = "http://172.18.127.52:8080/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base,)


prompt = '''
You are a meticulous AI Quality Analyst. Your role is to evaluate an AI-generated RESPONSE based on its adherence to a given INSTRUCTION. The evaluation must be performed against a CHECKLIST of criteria.

# Provided Information
INSTRUCTION: The original request given to the AI.
<INSTRUCTION>

RESPONSE: The AI's generated output that you must evaluate.
<RESPONSE>

CHECKLIST: The list of criteria the RESPONSE must satisfy.
<CHECKLIST>

# Task & Instructions
For each question in the CHECKLIST, you must perform the following steps:
1. Provide a concise analysis of how the RESPONSE performs against each question in the CHECKLIST.
2. Your analysis must justify your final verdict by referencing specific parts of the RESPONSE and the INSTRUCTION.
3. Conclude with a definitive Yes or No answer. Yes indicates the RESPONSE successfully meets the criterion; No indicates it does not.


# Response Format
Your response must strictly adhere to the following format, without any introductory or concluding remarks.
Question 1: [Insert the first question from the CHECKLIST here]
[Your detailed analysis of the RESPONSE against the first criterion, including specific references or quotes to justify your conclusion.]
<Answer>
Yes or No answer here, where 'yes' means the RESPONSE satisfies the criterion.
</Answer>

Question 2: [Insert the second question from the CHECKLIST here]
[Your detailed analysis of the RESPONSE against the second criterion, including specific references or quotes to justify your conclusion.]
<Answer>
Yes or No answer here, where 'yes' means the RESPONSE satisfies the criterion.
</Answer>

...and so on for all questions in the CHECKLIST.
'''
prompt = prompt.strip()


def compute_score(prompt_strs, solution_strs, extra_infos, llm, llm_tokenizer):
    messages = [
        [{"role": "user", "content": prompt.replace("<INSTRUCTION>", prompt_str).replace("<RESPONSE>", solution_str).replace("<CHECKLIST>", "\n".join(extra_info['checklist']))}] for prompt_str, solution_str, extra_info in zip(prompt_strs, solution_strs, extra_infos)
    ]

    prompts = [
        llm_tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ) for message in messages
    ]

    # import pdb; pdb.set_trace()
    print(f"Evaluating checklist with {len(prompts)} prompts using model {llm}...")
    wake = requests.post("http://172.18.127.52:8080/wake_up")
    print(wake)
    eval_batch_size = 1000
    responses = []
    for i in tqdm(range(0, len(prompts), eval_batch_size)):
        batch_prompts = prompts[i:i + eval_batch_size]
        batch_responses = client.completions.create(
            model=llm,
            prompt=batch_prompts,
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
        ).choices
        responses.extend([response.text.strip() for response in batch_responses])

    # responses = client.completions.create(
    #     model=llm,
    #     prompt=prompts,
    #     max_tokens=1024,
    #     temperature=0.0,
    #     top_p=1.0,
    # ).choices
    sleep = requests.post("http://172.18.127.52:8080/sleep")
    print(sleep)

    scores = []
    for i, response in enumerate(responses):
        checklist = extra_infos[i].get("checklist", [])
        all_score = []
        for j, question in enumerate(checklist):
            match = re.search(rf"Question\s*{j+1}:.*?<Answer>\s*(.*?)\s*</Answer>", response, re.IGNORECASE | re.DOTALL)
            if match:
                all_score.append(1.0 if "yes" in match.group(1).strip().lower() else 0.0)
            else:
                all_score.append(0.0)  # Default to 0.0 if no match found
        score = {
            "score": sum(all_score) / len(all_score) if all_score else 0.0,
            "score_for_checklist": '\n'.join([str(s) for s in all_score]),
            "evaluation": response,
        }
        scores.append(score)
    
    
    return scores