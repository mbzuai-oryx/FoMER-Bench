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

os.environ["HF_TOKEN"] = ""
MODEL_PATH = "gemini-2.5-pro"
genai.configure(api_key="")


SYSTEM_PROMPT = """You are a highly analytical assistant specialized in embodied reasoning.
                Come to the conclusion after methodically analyzing the provided frames, questions, and options.
                Then evaluate the options based on your reasoning and come to a final conclusion.

                Answer the question in the following format:


                <think>\nProvide a detailed step-by-step explanation that logically leads to the answer\n</think>\n\n<answer>\nThe final answer selected from the given options, without any explanation.\n</answer>

                """

SYSTEM_PROMPT_OPEN =  """You are a highly analytical assistant specialized in embodied reasoning.
                Come to the conclusion after methodically analyzing the provided frame, and the question.
                Then evaluate possible perspectives and come to a final conclusion.

                Answer the question in the following  format:

                <think>\nProvide a detailed step-by-step explanation that logically leads to the answer\n</think>\n\n<answer>\nThe final answer\n</answer>

                """

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']



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

def infer(clip_data: Dict[str, Any], model: genai.GenerativeModel) -> Dict[str, Any]:
    question = clip_data['question']
    choices = clip_data['choices']
    frames = [to_pil(img) for img in clip_data['images']]
    
    try:

        if choices and choices[0] != "":
            prompt_parts = [
            f'''
                Question : {question}
                Options :
                    {"/n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices)])}
                ''',
                *frames
            ]
        else:
            prompt_parts = [
            f'''
                Question : {question}
                ''',
                *frames
        ]
        responses = model.generate_content(prompt_parts,
                                           safety_settings={
                                'HATE': 'BLOCK_NONE',
                                'HARASSMENT': 'BLOCK_NONE',
                                'SEXUAL' : 'BLOCK_NONE',
                                'DANGEROUS' : 'BLOCK_NONE'
                            })
        generated_text = responses.text.strip()

        try:
            if  "</think>" in generated_text:
                think_part, answer_part = generated_text.split("</think>", 1)
                think_part = think_part.replace("<think>", "").strip()
                answer_part = answer_part.replace("<answer>", "").replace("</answer>", "").strip()
                clip_data["generated_reasoning"] = think_part
                clip_data["generated_answer"] = answer_part
        except ValueError:
            print(f"Error splitting answer: {generated_text}")
            clip_data["generated_reasoning"] = "Error in reasoning"
            clip_data["generated_answer"] = generated_text

        
    except Exception as e:
        print(f"Error processing {clip_data['id']}: {str(e)}")
        print("-" * 20)
        

    return clip_data


def process_split(dataset: List[Dict[str, Any]], output_dir: str, split: str):

    # limit the dataset size for testing
    # dataset = dataset.select(range(5))

    if split in ["nyuvinn", "roboset", "recon"]:
        sys_prompt = SYSTEM_PROMPT_OPEN
    else:
        sys_prompt = SYSTEM_PROMPT
    model = genai.GenerativeModel(
        model_name=MODEL_PATH,
        system_instruction=sys_prompt
    )



    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"results.jsonl")

    with open(output_path, "w") as out_file:
        for clip_data in tqdm(dataset, desc="Processing dataset"):
            result = infer(clip_data, model)
            # drop images to reduce file size
            result.pop("images", None)
            out_file.write(json.dumps(result) + "\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--split", type=str, default="agibot_world", help="Dataset split to process (train/val/test)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving results")

    split = parser.parse_args().split
    output_dir = parser.parse_args().output_dir

    dataset = load_dataset("Dinura/FoMER", split=split)
    process_split(dataset, output_dir, split)

