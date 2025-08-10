from ast import literal_eval
import functools
import json
import os
import random
import shutil
from tqdm import tqdm
# Scienfitic packages
import torch
torch.set_grad_enabled(False)

# Utilities

from general_utils import (
  DeSTAModelAndTokenizer
)

from patchscopes_utils import *

import argparse
import csv

model_to_hook = {
    "meta-llama/Meta-Llama-3-8B-Instruct": set_hs_patch_hooks_llama,
    "DeSTA-ntu/DeSTA2-8B-beta": set_hs_patch_hooks_llama,
}

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', '-a', type=str, choices=['Emotion', 'Language', 'Gender', 'Animal'], help="The attribute to probe", required=True)
    parser.add_argument('--output_dir', '-o', type=str, default="./results/")
    parser.add_argument('--cache_dir', '-c', type=str, default="./cache")
    parser.add_argument('--hf_token', type=str)
    parser.add_argument('--use_user_prompt', '-u', action='store_true', help="Use user prompt")
    parser.add_argument('--use_option', '-t', action='store_true')
    parser.add_argument('--reverse', '-r', action='store_true')
    parser.add_argument('--from_target_end', '-f', action='store_true')
    parser.add_argument('--use_is', '-i', action='store_true', help="Use is in the user prompt")
    parser.add_argument('--no_audio', '-n', action='store_true', help="Include audio or not")
    return parser.parse_args()

def read_json_file(file_path):
    """Reads a JSON file and returns the data."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")

def save_dict_to_json(dictionary, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dictionary, json_file, ensure_ascii=False, indent=4)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Failed: {e}")

def check_and_create_folder(folder_path):
    """
    Check if a folder exists, and create it if it doesn't.

    Args:
        folder_path (str): The path to the folder to check or create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def read_csv_file(file_path):
    data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return data

def write_to_csv(file_path, data, fieldnames):
    try:
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()

            for row in data:
                writer.writerow(row)
        print(f"Data successfully written to '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def list_wav_files(folder_path):
    """
    Recursively find all .wav files under folder_path.

    Args:
        folder_path (str): Path to the root folder.

    Returns:
        list of str: Full paths to all .wav files.
    """
    wav_files = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, fname))
    return wav_files
        
if __name__ == "__main__":
    args = get_args_parser()
    check_and_create_folder(args.output_dir)
    
    model_name = "DeSTA-ntu/DeSTA2-8B-beta"
    sos_tok = False

    desta_mt = DeSTAModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        cache_dir=args.cache_dir,
        device="cuda:0"
    )
    desta_mt.set_hs_patch_hooks = model_to_hook[model_name]
    desta_mt.model.eval()

    if not args.reverse:
        probing_prompts = {
            "Emotion": "The speaker's emotion",
            "Language": "The speech's spoken language",
            "Gender": "The speaker's gender",
            "Animal": "The sound file's animal",
        }
        
        position_source = -1
        position_target = -1
    else:
        probing_prompts = {
            "Emotion": "The emotion of the speaker",
            "Language": "The spoken language of the speech",
            "Gender": "The gender of the speaker",
            "Animal": "The animal in the sound file",
        }
        if args.from_target_end:
            position_source = -1
            position_target = -1
        else:
            position_source = -4 if args.attribute != "Animal" else -5
            position_target = -4 if args.attribute != "Animal" else -5
    
    if args.use_is:
        probing_prompts = {k: v.strip() + " is" for k, v in probing_prompts.items()}
    
    
    if args.use_option:
        user_prompts = {
            "Emotion": "What is the emotion of the speaker in the speech? Possible options: angry, disgust, fear, happy, sad.",
            "Language": "What is the language spoken in the speech? Possible options: English, German, Spanish, French, Italian, Chinese, Japanese, Korean.",
            "Gender": "What is the gender of the speaker in the speech? Possible options: male, female.",
            "Animal": "What animal makes the sound? Possible options: dog, cat, pig, cow, frog, hen, rooster, sheep, crow.",
        }
    else:
        user_prompts = {
            "Emotion": "What is the emotion of the speaker in the speech?",
            "Language": "What is the language spoken in the speech?",
            "Gender": "What is the gender of the speaker in the speech?",
            "Animal": "What animal makes the sound?",
        }
    probing_prompt = probing_prompts[args.attribute]
    print("Using probing prompt: ", probing_prompt)
    
    if args.use_user_prompt:
        user_prompt = user_prompts[args.attribute]
        print("Using user prompt: ", user_prompt)
    else:
        user_prompt = None
    layer_source = -1
    layer_target = -1

    option_dict = {
        "Emotion": ['angry', 'disgust', 'fear', 'happy', 'sad'],
        "Language": ["English", "German", "Spanish", "French", "Italian", "Chinese", "Japanese", "Korean"],
        "Gender": ["male", "female"],
        "Animal": ["dog", "cat", "pig", "cow", "frog", "hen", "rooster", "sheep", "crow"]
    }
    
    options = option_dict[args.attribute]
        
    # TODO: Fill in path to your dataset
    track_dir = f"./{args.attribute}"
    metadata = read_csv_file(os.path.join(track_dir, "metadata.csv"))

    outputs = {}
    outputs['source_prompt'] = probing_prompt
    outputs['target_prompt'] = probing_prompt
    outputs['source_user_prompt'] = user_prompt
    outputs['target_user_prompt'] = user_prompt
    outputs['position_source'] = position_source
    outputs['position_target'] = position_target
    outputs['sample_results'] = {}
    
    print("Position source: ", position_source)
    print("Position target: ", position_target)
    
    for sample in tqdm(metadata):
        audio_source = os.path.join(track_dir, sample['file_name']) if not args.no_audio else None

        audio_target = audio_source
        attribute_label = sample['attribute_label']
        
        output_item = {
            "audio_source": audio_source,
            "audio_target": audio_target,
            "attribute_label": attribute_label
        }
        
        output_dict = inspect_desta_loop_across_layers(
                    desta_mt,
                    prompt_source=probing_prompt,
                    prompt_target=probing_prompt,
                    user_prompt_source=user_prompt,
                    user_prompt_target=user_prompt,
                    layer_source=layer_source,
                    layer_target=layer_target,
                    position_source=position_source,
                    position_target=position_target,
                    module="hs",
                    generation_mode=False,
                    max_gen_len=20,
                    temperature=None,
                    audio_source=audio_source,
                    audio_target=audio_target,
                    patch_from_audio_end=False,
                    patch_to_audio_end=False,
                    loop_layer_source=True,
                    loop_layer_target=False,
                    options=options
                )
        
        output_item['layer_results'] = output_dict
        
        outputs['sample_results'][sample['file_name']] = output_item

    if args.use_user_prompt:
        if args.use_option:
            json.dump(outputs, open(os.path.join(args.output_dir, f"{args.attribute}_logit_with_user_prompt_options.json"), "w"), indent=4)
        else:
            json.dump(outputs, open(os.path.join(args.output_dir, f"{args.attribute}_logit_with_user_prompt.json"), "w"), indent=4)
    
    else:
        json.dump(outputs, open(os.path.join(args.output_dir, f"{args.attribute}_logit.json"), "w"), indent=4)