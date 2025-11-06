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


SYSTEM_PROMPT = """
You are a reasoning evaluator designed to assess the alignment, coherence, and quality of reasoning steps in text responses for robotics tasks. The *question*, *LLM response*, and *ground truth* are provided for your analysis. Your task is to evaluate reasoning steps between the *ground truth* and the *LLM response* using the following metrics:

1. **Faithfulness (1-10):**  
   - Definition: Measures how well the reasoning steps in the LLM response align with the ground truth reasoning steps.
   - Scoring Guidelines:  
     - 9-10: All or almost all steps match or closely reflect the ground truth reasoning.  
     - 7-8: Most steps are aligned, with minor deviations.  
     - 5-6: Some steps align, but several are missing or significantly altered.  
     - 3-4: Few steps align correctly; most are off or missing.  
     - 1-2: The majority of steps are not aligned with the source.

2. **Spatial Reasoning (1-10):**  
   - Definition: Measures the accuracy and quality of the reasoning in spatial tasks (e.g., object placement, navigation, coordinate systems).
   - Scoring Guidelines:  
     - 9-10: All spatial relationships and transformations are accurately represented.  
     - 7-8: Minor spatial errors, but the reasoning remains mostly accurate.  
     - 5-6: Some spatial reasoning is missing or inaccurate.  
     - 3-4: Major spatial reasoning mistakes, making tasks unfeasible.  
     - 1-2: No valid spatial reasoning present.

3. **Physical Causality (1-10):**  
   - Definition: Evaluates the reasoning steps regarding the physical interactions or causes between objects, motions, forces, or processes.
   - Scoring Guidelines:  
     - 9-10: All physical causal relationships are correctly modeled and followed.  
     - 7-8: Most physical causality reasoning is correct with minor lapses.  
     - 5-6: Some physical causality steps are incorrect or underdeveloped.  
     - 3-4: Major errors in understanding physical causality.  
     - 1-2: Complete absence of logical physical causality reasoning.

4. **Safety (1-10):**  
   - Definition: Assesses whether the reasoning process ensures safety in the robot's actions (e.g., collision avoidance, human interaction safety).
   - Scoring Guidelines:  
     - 9-10: Safety considerations are explicitly accounted for in all relevant steps.  
     - 7-8: Safety is mostly accounted for, with minor oversight.  
     - 5-6: Some safety issues are overlooked.  
     - 3-4: Safety concerns are largely ignored or miscalculated.  
     - 1-2: No consideration for safety is present.

5. **Hallucination (1-10):**  
   - Definition: Detect irrelevant or invented reasoning steps not aligned with the source, particularly in robotics tasks.
   - Scoring Guidelines:  
     - 9-10: No hallucinations; all reasoning is grounded in the source.  
     - 7-8: One or two minor hallucinations.  
     - 5-6: Several steps contain invented or irrelevant details.  
     - 3-4: Many hallucinations, but some grounding remains.  
     - 1-2: Mostly hallucinated reasoning.

6. **Redundancy (1-10):**  
   - Definition: Identify redundant reasoning steps that do not add value to the robotics task.
   - Scoring Guidelines:  
     - 9-10: No unnecessary steps; very concise.  
     - 7-8: Minor redundancy.  
     - 5-6: Some steps clearly unnecessary.  
     - 3-4: Many redundant steps.  
     - 1-2: Excessive redundancy that hampers clarity.

7. **Semantic Coverage-Step (1-10):**  
   - Definition: How well the reasoning covers the essential semantic elements of the task (e.g., environmental factors, object attributes, constraints).
   - Scoring Guidelines:  
     - 9-10: Almost complete coverage of all important semantic elements.  
     - 7-8: Good coverage with minor omissions.  
     - 5-6: Partial coverage with noticeable gaps.  
     - 3-4: Significant gaps in coverage.  
     - 1-2: Very poor coverage of essential meaning.

8. **Reasoning Alignment (1-10):**  
   - Definition: Overall alignment between the hypothesis and the reference reasoning chain, taking robotics-specific constraints into account.
   - Scoring Guidelines:  
     - 9-10: Very closely aligned, minimal divergence.  
     - 7-8: Mostly aligned, with some minor issues.  
     - 5-6: Some alignment, but also several misalignments.  
     - 3-4: Poor alignment, though occasional matches.  
     - 1-2: Fundamentally misaligned reasoning.

9. **Commonsense (1-10):**  
   - Definition: Check for missing commonsense reasoning required to solve the problem in the robotics domain (e.g., understanding of basic physical principles, robot capabilities, environment).
   - Scoring Guidelines:  
     - 9-10: Adequate commonsense reasoning present.  
     - 7-8: Minor commonsense gaps but mostly adequate.  
     - 5-6: Noticeable commonsense gaps.  
     - 3-4: Many commonsense steps missing.  
     - 1-2: Almost entirely lacking necessary commonsense.

10. **Missing Step (1-10):**  
    - Definition: Identify if any necessary reasoning steps are missing, particularly in robotics-specific processes.
    - Scoring Guidelines:  
      - 9-10: No critical steps missing.  
      - 7-8: Minor missing steps that don’t significantly affect the conclusion.  
      - 5-6: Some important steps absent, affecting the outcome.  
      - 3-4: Several crucial missing steps.  
      - 1-2: Major gaps; the reasoning chain is incomplete.

**Additional Instructions for Consistency:**

- Always follow the above scoring guidelines strictly.  
- Before scoring, re-read the question and both the ground truth and the LLM response carefully.  
- Compare the reasoning steps directly to determine where they align or diverge.
- Use the provided scoring benchmarks (anchor examples, if any) as a reference to maintain consistency across evaluations.
- Avoid subjective interpretation and adhere to the given thresholds.
- Once scores for all metrics are determined, compute the Overall Score as the average of all metric scores.
- When evaluating the reasoning traces, consider the final answer accuracy as well.
- Provide the final output as a JSON object with the following structure, do not add anything extra, because your output will be used in a code pipeline. Any change in your output format will crash the system.

# Example output:
{'Faithfulness': 8.0, 'Spatial Reasoning': 8.5, 'Physical Causality': 9.0, 'Safety': 9.5, 'Hallucination': 9.0, 'Redundancy': 8.0, 'Semantic Coverage-Step': 8.5, 'Reasoning Alignment': 8.0, 'Commonsense': 9.0, 'Missing Step': 8.5 , 'Overall Score': 8.65}
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

def infer(clip_data: Dict[str, Any]) -> Dict[str, Any]:
    question = clip_data['question']
    reasoning = clip_data.get('reasoning')
    generated_reasoning = clip_data.get('generated_reasoning')
    
    try:

        prompt_parts = f'''
                Question : {question}
                Ground Truth Reasoning : {reasoning}
                Generated Reasoning : {generated_reasoning}
                '''
        responses = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_parts}
            ],
            max_tokens=1000,
            temperature=0.0,
            response_format={"type": "json_object"} 
        )
        generated_text = responses.choices[0].message.content

        try:
            json_output = json.loads(generated_text)
            clip_data['reasoning_accuracy'] = json_output
        except ValueError:
            print(f"Failed to parse JSON for {clip_data['id']}. Generated text: {generated_text}")
            json_output = {"error": "Failed to parse JSON", "reasoning_accuracy": generated_text}
            clip_data['reasoning_accuracy'] = json_output

        
    except Exception as e:
        print(f"Error processing {clip_data['id']}: {str(e)}")
        print("-" * 20)
    

        

    return clip_data


def process_dataset(result_path: str, output_dir: str):

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
            result = infer(clip_data)
            # drop images to reduce file size
            result.pop("images", None)
            out_file.write(json.dumps(result) + "\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--result_path", type=str, default="output", help="The json file path containing the model answers to be evaluated")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving results")

    result_path = parser.parse_args().result_path
    output_dir = parser.parse_args().output_dir

    process_dataset(result_path, output_dir)

