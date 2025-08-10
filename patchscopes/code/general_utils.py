# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import re

import torch
import transformers
from desta import DestaModel
import librosa


class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      use_fast=True,
      device="cuda",
      cache_dir=None,
      ):
    if tokenizer is None:
      assert model_name is not None
      if cache_dir is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
      else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast, cache_dir=cache_dir
        )
    if model is None:
      assert model_name is not None
      if cache_dir is None:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype
            )
      else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype, cache_dir=cache_dir
            )
      if device is not None:
        model.to(device)
      set_requires_grad(False, model)
      model.eval()
    self.tokenizer = tokenizer
    self.model = model
    self.device = device
    self.layer_names = [
        n
        for n, _ in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )

class DeSTAModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      hf_token,
      model_name="DeSTA-ntu/DeSTA2-8B-beta",
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      use_fast=True,
      device="cuda",
      cache_dir="./cache"
      ):
    
    if model is None:
      model = DestaModel.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir).to(device)
    if device is not None:
      model.to(device)
    
    set_requires_grad(False, model.speech_perception.whisper)
    set_requires_grad(False, model.speech_perception.connector)
    set_requires_grad(False, model.llama)
    model.eval()

    # print("Success")

    self.tokenizer = model.tokenizer
    self.model = model.llama
    self.speech_perception = model.speech_perception
    self.device = device
    self.layer_names = [
        n
        for n, _ in self.model.named_modules()
        if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)


  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"DeSTAModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )
  
  def process_text(self, messages, audio_path, transcription, remove_last_eot=True):
      if remove_last_eot:
        context = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        context = self.tokenizer.decode(context[:-1]) # Remove last eot since we don't need it
      else:
        context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

      left_text, right_text = context.split(audio_path)
      right_text = transcription + right_text
      
      audio_position = len(self.tokenizer.tokenize(left_text))
      context = left_text + right_text

      inputs = self.tokenizer(context, return_tensors="pt")

      return inputs, audio_position
  
  def process_text_without_audio(self, messages, remove_last_eot=True):
      if remove_last_eot:
        context = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        context = self.tokenizer.decode(context[:-1]) # Remove last eot since we don't need it
      else:
        context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
      inputs = self.tokenizer(context, return_tensors="pt")

      return inputs
  
  def prepare_llm_input(self, input_ids, attention_mask, audio_position, audio_features, return_audio_position=True):
      input_ids = input_ids.to(self.model.device)
      attention_mask = attention_mask.to(self.model.device)
      audio_features = audio_features.to(self.model.device)
      audio_feature_length = audio_features.size(1)

      inputs_embeds = self.model.model.embed_tokens(input_ids) # [bs, seq_len, hidden_size]


      inputs_embeds = torch.cat([inputs_embeds[0, :audio_position], audio_features[0, :], inputs_embeds[0, audio_position:]], dim=0)
      attention_mask = torch.cat([attention_mask[0, :audio_position], torch.ones([ audio_feature_length], dtype=torch.long, device=self.model.device), attention_mask[0, audio_position:]], dim=0)

      inputs_embeds = inputs_embeds.to(self.model.dtype)
      attention_mask = attention_mask.to(self.model.dtype)
      if return_audio_position:
          return inputs_embeds.unsqueeze(0), attention_mask.unsqueeze(0), (audio_position, audio_feature_length + audio_position - 1) # (start, end) for audio position, may be useful for patching
      else:
          return inputs_embeds.unsqueeze(0), attention_mask.unsqueeze(0)

  def prepare_llm_input_without_audio(self, input_ids, attention_mask):
      input_ids = input_ids.to(self.model.device)
      attention_mask = attention_mask.to(self.model.device)

      inputs_embeds = self.model.model.embed_tokens(input_ids) # [bs, seq_len, hidden_size]

      inputs_embeds = torch.cat([inputs_embeds[0]], dim=0)
      attention_mask = torch.cat([attention_mask[0]], dim=0)
      inputs_embeds = inputs_embeds.to(self.model.dtype)
      attention_mask = attention_mask.to(self.model.dtype)
      return inputs_embeds.unsqueeze(0), attention_mask.unsqueeze(0)
  
  def load_audio(self, messages):
      audio_path = None
      for message in messages:
          if message["role"] == "audio" and audio_path is not None:
              raise ValueError("Multiple audio file paths found in messages. We only support one audio file per message at this moment.")
          if message["role"] == "audio":
              audio_path = message["content"]
      if audio_path is None:
          raise ValueError("No audio file path found in messages")
      audio, ori_sr = librosa.load(audio_path)
      audio = librosa.resample(audio, orig_sr=ori_sr, target_sr=16000)
      input_features = self.speech_perception.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

      return audio_path, input_features


def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)
