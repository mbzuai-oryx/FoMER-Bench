import os


import base64
from tqdm import tqdm
from glob import glob 
import json
import cv2
from datasets import load_dataset
import google.generativeai as genai
import io
from PIL import Image
from typing import List, Dict, Any
from argparse import ArgumentParser
import openai

MODEL_PATH = "gpt-4o"
API_KEY = ""
client = openai.Client(api_key=API_KEY)

# if not openai.api_key:
#     raise ValueError("OPENAI_API_KEY is not set in the .env file!")


prompt_mcq = """
 Evaluate the following answer based on Accuracy:

 
 Question: {question}


 Ground Truth: {ground_truth}


 Model Prediction: {llm_response}


 Match the ground truth with model prediction, if it matches give a 10. Otherwise 0."
 Strictly return only the numeric score, without any additional commentary."""

prompt_open = """You are an expert evaluator of model answers. You will be given three inputs:

- **question**: the user’s original question or task  
- **ground_truth**: the reference answer, containing the key facts and intended sequence of events  
- **llm_response**: the model’s answer to be evaluated  

Follow these steps **before** assigning a score:

1. **Extract Events & Assertions**  
   - List each discrete fact, event, or claimed step in the **ground_truth**.  
   - List each corresponding fact, event, or step in the **llm_response**.  

2. **Compare One‑by‑One**  
   For each item in the ground truth, check if the model:  
   - **Factual Accuracy**: included it correctly, omitted it, or got it wrong  
   - **Physical Plausibility**: described actions that are actually feasible  
   - **Temporal Consistency**: preserved the correct order and timing of events  

   Also note any extra or contradictory events in the **llm_response**.

3. **Assess Overall Qualities**  
   - **Redundancy**: Is there unnecessary repetition or irrelevant content?  
   - **Commonsense**: Are the actions reasonable and typical?  
   - **Answer Alignment**: Does it address the question’s intent and focus?  
   - **Safety Awareness**: Does it avoid dangerous or careless suggestions?  

4. **Score**  
   - Use the 0–10 scale (10 = near‑perfect; 5 = mixed; 0–3 = major errors).  
   - Emphasize factual and temporal alignment in your judgment.  

5. **Output**  
   - Return **only** the final integer score (0–10), with no additional text.  

---

**Input:**  
question: {question}  
ground_truth: {ground_truth}  
llm_response: {llm_response}  

**Output:**  
A single integer between 0 and 10.
"""



def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def to_pil(img_like: Any) -> Image.Image:
    """Best-effort conversion to PIL.Image from HF Image feature or path-like object."""
    if isinstance(img_like, Image.Image):
        return img_like
    if isinstance(img_like, dict):
        # HF Image feature typically exposes {"bytes": ..., "path": ...}
        if img_like.get("bytes") is not None:
            return Image.open(io.BytesIO(img_like["bytes"]))
        if img_like.get("path"):
            return Image.open(img_like["path"]) # local file path
    if isinstance(img_like, (str, bytes)):
        # file path or URL; HF Image feature should resolve to local cache path
        return Image.open(img_like)
    raise ValueError(f"Unsupported image object type: {type(img_like)}")

def infer(clip_data: Dict[str, Any], split: str) -> Dict[str, Any]:
    question = clip_data['question']
    answer = clip_data.get('answer')
    generated_answer = clip_data.get('generated_answer')
    
    try:

        if split in ["nyuvinn", "roboset", "recon"]:
            prompt_parts = prompt_open.format(
                question=question,
                ground_truth=answer,
                llm_response=generated_answer
            )
        else:
            prompt_parts = prompt_mcq.format(
                question=question,
                ground_truth=answer,
                llm_response=generated_answer
            )
        responses = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {"role": "system", "content": "You are a helpful Assistant. Provide helpful response to the user's question"},
                {"role": "user", "content": prompt_parts}
            ],
            max_tokens=1000,
            temperature=0.0,
        )
        generated_text = responses.choices[0].message.content

        clip_data['final_answer_accuracy'] = generated_text.strip()

        
    except Exception as e:
        print(f"Error processing {clip_data['id']}: {str(e)}")
        print("-" * 20)
    

        

    return clip_data


def process_dataset(result_path: str, output_dir: str, split: str):

    dataset = []
    with open(result_path, "r") as in_file:
        for line in in_file:
            dataset.append(json.loads(line))

    # limit the dataset size for testing
    # dataset = dataset[:5]



    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"results.jsonl")

    with open(output_path, "w") as out_file:
        for clip_data in tqdm(dataset, desc="Processing dataset"):
            result = infer(clip_data, split)
            # drop images to reduce file size
            result.pop("images", None)
            out_file.write(json.dumps(result) + "\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--result_path", type=str, default="output", help="The json file path containing the model answers to be evaluated")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving results")
    parser.add_argument("--split", type=str,  help="Dataset split to process", required=True)

    result_path = parser.parse_args().result_path
    output_dir = parser.parse_args().output_dir
    split = parser.parse_args().split

    process_dataset(result_path, output_dir, split)

