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

import numpy as np
import torch
import tqdm
from general_utils import decode_tokens
from general_utils import make_inputs

from tqdm import tqdm

# ##############
#
# Hooks
#
# ##############


def set_hs_patch_hooks_neox(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """Neox patch hooks."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_

    def post_hook(module, input, output):
      if "skip_ln" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name:
          output[0][position_] = hs_
        else:
          output[0][0, position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    if patch_input:
      hooks.append(
          model.gpt_neox.layers[i].register_forward_pre_hook(
              patch_hs(
                  f"patch_hs_{i}",
                  hs_patch_config[i],
                  patch_input,
                  generation_mode,
              )
          )
      )
    else:
      # when patching a last-layer representation to the last layer of the
      # same model, the final layer norm is not needed because it was already
      # applied (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.gpt_neox.layers) - 1:
        hooks.append(
            model.gpt_neox.final_layer_norm.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        hooks.append(
            model.gpt_neox.layers[i].register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )

  return hooks


def set_hs_patch_hooks_llama(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """Llama patch hooks."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_

    def post_hook(module, input, output):
      if "skip_ln" in name or "mlp" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name or "mlp" in name:
          output[0][position_] = hs_
        else:
          output[0][0, position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    patch_hook = patch_hs(
        f"patch_{module}_{i}",
        position_hs=hs_patch_config[i],
        patch_input=patch_input,
        generation_mode=generation_mode,
    )
    if patch_input:
      if module == "hs":
        hooks.append(
            model.model.layers[i].register_forward_pre_hook(patch_hook)
        )
      elif module == "mlp":
        hooks.append(
            model.model.layers[i].mlp.register_forward_pre_hook(patch_hook)
        )
      elif module == "attn":
        hooks.append(
            model.model.layers[i].self_attn.register_forward_pre_hook(
                patch_hook
            )
        )
      else:
        raise ValueError("Module %s not supported", module)
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already applied
      # (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.model.layers) - 1 and module == "hs":
        hooks.append(
            model.model.norm.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        if module == "hs":
          hooks.append(model.model.layers[i].register_forward_hook(patch_hook))
        elif module == "mlp":
          hooks.append(
              model.model.layers[i].mlp.register_forward_hook(patch_hook)
          )
        elif module == "attn":
          hooks.append(
              model.model.layers[i].self_attn.register_forward_hook(patch_hook)
          )
        else:
          raise ValueError("Module %s not supported", module)

  return hooks

def set_hs_patch_hooks_llama_forward_add(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
    forward_add_coefficient=1,  # whether to use forward add instead of forward hook
):
  """Llama patch hooks."""
#   print(forward_add_coefficient)
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] += forward_add_coefficient * hs_

    def post_hook(module, input, output):
      if "skip_ln" in name or "mlp" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name or "mlp" in name:
          output[0][position_] += forward_add_coefficient * hs_
        else:
          output[0][0, position_] += forward_add_coefficient * hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    patch_hook = patch_hs(
        f"patch_{module}_{i}",
        position_hs=hs_patch_config[i],
        patch_input=patch_input,
        generation_mode=generation_mode,
    )
    if patch_input:
      if module == "hs":
        hooks.append(
            model.model.layers[i].register_forward_pre_hook(patch_hook)
        )
      elif module == "mlp":
        hooks.append(
            model.model.layers[i].mlp.register_forward_pre_hook(patch_hook)
        )
      elif module == "attn":
        hooks.append(
            model.model.layers[i].self_attn.register_forward_pre_hook(
                patch_hook
            )
        )
      else:
        raise ValueError("Module %s not supported", module)
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already applied
      # (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.model.layers) - 1 and module == "hs":
        hooks.append(
            model.model.norm.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        if module == "hs":
          hooks.append(model.model.layers[i].register_forward_hook(patch_hook))
        elif module == "mlp":
          hooks.append(
              model.model.layers[i].mlp.register_forward_hook(patch_hook)
          )
        elif module == "attn":
          hooks.append(
              model.model.layers[i].self_attn.register_forward_hook(patch_hook)
          )
        else:
          raise ValueError("Module %s not supported", module)

  return hooks


def set_hs_patch_hooks_llama_forward_average(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """Llama patch hooks."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = (input[0][0, position_] + hs_) / 2

    def post_hook(module, input, output):
      if "skip_ln" in name or "mlp" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name or "mlp" in name:
          output[0][position_] = (input[0][0, position_] + hs_) / 2
        else:
          output[0][0, position_] = (input[0][0, position_] + hs_) / 2

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    patch_hook = patch_hs(
        f"patch_{module}_{i}",
        position_hs=hs_patch_config[i],
        patch_input=patch_input,
        generation_mode=generation_mode,
    )
    if patch_input:
      if module == "hs":
        hooks.append(
            model.model.layers[i].register_forward_pre_hook(patch_hook)
        )
      elif module == "mlp":
        hooks.append(
            model.model.layers[i].mlp.register_forward_pre_hook(patch_hook)
        )
      elif module == "attn":
        hooks.append(
            model.model.layers[i].self_attn.register_forward_pre_hook(
                patch_hook
            )
        )
      else:
        raise ValueError("Module %s not supported", module)
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already applied
      # (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.model.layers) - 1 and module == "hs":
        hooks.append(
            model.model.norm.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        if module == "hs":
          hooks.append(model.model.layers[i].register_forward_hook(patch_hook))
        elif module == "mlp":
          hooks.append(
              model.model.layers[i].mlp.register_forward_hook(patch_hook)
          )
        elif module == "attn":
          hooks.append(
              model.model.layers[i].self_attn.register_forward_hook(patch_hook)
          )
        else:
          raise ValueError("Module %s not supported", module)

  return hooks

def set_hs_patch_hooks_gptj(
    model,
    hs_patch_config,
    module="hs",  # mlp, attn
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
  """GPTJ patch hooks."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need
  # to handle these cases in this call because this hook wraps the generation
  # call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation
  # if we are handling the initial input or a future step and thus don't know
  # if a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, input):
      # (batch, sequence, hidden_state)
      input_len = len(input[0][0])
      if generation_mode and input_len == 1:
        return
      for position_, hs_ in position_hs:
        input[0][0, position_] = hs_

    def post_hook(module, input, output):
      if "skip_ln" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[0])
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][0])

      if generation_mode and output_len == 1:
        return
      for position_, hs_ in position_hs:
        if "skip_ln" in name:
          output[0][position_] = hs_
        else:
          output[0][0, position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for i in hs_patch_config:
    if patch_input:
      hooks.append(
          model.transformer.h[i].register_forward_pre_hook(
              patch_hs(
                  f"patch_hs_{i}",
                  hs_patch_config[i],
                  patch_input,
                  generation_mode,
              )
          )
      )
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already applied
      # (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.transformer.h) - 1:
        hooks.append(
            model.transformer.ln_f.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        hooks.append(
            model.transformer.h[i].register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}",
                    hs_patch_config[i],
                    patch_input,
                    generation_mode,
                )
            )
        )

  return hooks


def remove_hooks(hooks):
  for hook in hooks:
    hook.remove()


# ##############
#
# Inspection
#
# ##############


def inspect(
    mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    generation_mode=False,
    max_gen_len=20,
    verbose=False,
    temperature=None,
):
  """Inspection via patching."""
  # adjust position_target to be absolute rather than relative
  inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
  if position_target < 0:
    position_target = len(inp_target["input_ids"][0]) + position_target

  # first run the the model on prompt_patch and get all hidden states.
  inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
  if verbose:
    print(
        "prompt_patch:",
        [mt.tokenizer.decode(x) for x in inp_source["input_ids"][0]],
    )

  hs_cache_ = []
  # We manually store intermediate states that the model API does not expose
  store_hooks = []
  if module == "mlp":

    def store_mlp_hook(module, input, output):
      hs_cache_.append(output[0])

    for layer in mt.model.model.layers:
      store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
  elif module == "attn":

    def store_attn_hook(module, input, output):
      hs_cache_.append(output[0].squeeze())

    for layer in mt.model.model.layers:
      store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))

  output = mt.model(**inp_source, output_hidden_states=True)
  if module == "hs":
    hs_cache_ = [
        output["hidden_states"][layer + 1][0] for layer in range(mt.num_layers)
    ]

  remove_hooks(store_hooks)
  # now do a second run on prompt, while patching
  # a specific hidden state from the first run.
  hs_patch_config = {
      layer_target: [(
          position_target,
          hs_cache_[layer_source][position_source],
      )]
  }

  if layer_source == layer_target == mt.num_layers - 1:
    skip_final_ln = True
  else:
    skip_final_ln = False
  patch_hooks = mt.set_hs_patch_hooks(
      mt.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )

  # Single prediction / generation
  if verbose:
    print(
        "prompt:", [mt.tokenizer.decode(x) for x in inp_source["input_ids"][0]]
    )
    print(
        f"patching position {position_target} with the hidden state from layer"
        f" {layer_source} at position {position_source}."
    )
  if generation_mode:
    # Checking if should perform temperature sampling, to allow smoother
    # non-repeating long outputs.
    if temperature:
      output_toks = mt.model.generate(
          inp_target["input_ids"],
          max_length=len(inp_target["input_ids"][0]) + max_gen_len,
          # pad_token_id=mt.model.generation_config.eos_token_id,
          pad_token_id=mt.model.config.eos_token_id,
          temperature=temperature,
          do_sample=True,
          top_k=0,
      )[0][len(inp_target["input_ids"][0]) :]
    else:
      # print(mt.model.config.eos_token_id)
      output_toks = mt.model.generate(
          inp_target["input_ids"],
          max_length=len(inp_target["input_ids"][0]) + max_gen_len,
          # pad_token_id=mt.model.generation_config.eos_token_id,
          pad_token_id=mt.model.config.eos_token_id
      )[0][len(inp_target["input_ids"][0]) :]

    output = mt.tokenizer.decode(output_toks)
    if verbose:
      print(
          "generation with patching: ",
          [mt.tokenizer.decode(x) for x in output_toks],
      )
  else:
    output = mt.model(**inp_target)
    answer_prob, answer_t = torch.max(
        torch.softmax(output.logits[0, -1, :], dim=0), dim=0
    )
    output = decode_tokens(mt.tokenizer, [answer_t])[0], round(
        answer_prob.cpu().item(), 4
    )
    if verbose:
      print("prediction with patching: ", output)

  # remove patching hooks
  remove_hooks(patch_hooks)

  return output

def inspect_desta(
    desta_mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    generation_mode=False,
    max_gen_len=20,
    verbose=False,
    temperature=None,
    audio_source=None, # Path to the audio file (source of patching)
    audio_target=None, # Path to the audio file (target of patching)
    patch_from_audio_end=False, # whether to patch from the end of the audio position
    patch_to_audio_end=False, # whether to patch to the end of the audio position
    user_prompt_source=None, # User prompt for the source
    user_prompt_target=None, # User prompt for the target
):
  """Inspection via patching."""
  if audio_source is not None:
    messages = [
                  {"role": "audio", "content": audio_source},
                  *([{"role": "user", "content": user_prompt_source}] if user_prompt_source is not None else []),
                  {"role": "assistant", "content": prompt_source}
              ]
    audio_source, input_features = desta_mt.load_audio(messages)
    transcription, audio_features = desta_mt.speech_perception.generate(input_features)
    inputs, audio_position = desta_mt.process_text(messages, audio_source, transcription, remove_last_eot=True)
    
    inputs_embeds_source, attention_mask_source, (audio_start_source, audio_end_source) = desta_mt.prepare_llm_input(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, 
            audio_position=audio_position,
            audio_features=audio_features
        )
    
  else:
    messages = [
                  *([{"role": "user", "content": user_prompt_source}] if user_prompt_source is not None else []),
                  {"role": "assistant", "content": prompt_source}
              ]
    inputs = desta_mt.process_text_without_audio(messages, remove_last_eot=True)
    inputs_embeds_source, attention_mask_source = desta_mt.prepare_llm_input_without_audio(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

  
  
  # first run the the model on prompt_patch and get all hidden states.

  hs_cache_ = []
  # We manually store intermediate states that the model API does not expose
  store_hooks = []
  if module == "mlp":

    def store_mlp_hook(module, input, output):
      hs_cache_.append(output[0])

    for layer in desta_mt.model.model.layers:
      store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
  elif module == "attn":

    def store_attn_hook(module, input, output):
      hs_cache_.append(output[0].squeeze())

    for layer in desta_mt.model.model.layers:
      store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))

  output = desta_mt.model(inputs_embeds=inputs_embeds_source, attention_mask=attention_mask_source, output_hidden_states=True)
  if module == "hs":
    hs_cache_ = [
        output["hidden_states"][layer + 1][0] for layer in range(desta_mt.num_layers)
    ]

  remove_hooks(store_hooks)
  
  # now do a second run on prompt, while patching
  # a specific hidden state from the first run.

  if audio_target is not None: # for logit lens case
    messages = [
                  {"role": "audio", "content": audio_target},
                  *([{"role": "user", "content": user_prompt_target}] if user_prompt_target is not None else []),
                  {"role": "assistant", "content": prompt_target}
              ]
    audio_target, input_features = desta_mt.load_audio(messages)
    transcription, audio_features = desta_mt.speech_perception.generate(input_features)
    inputs, audio_position = desta_mt.process_text(messages, audio_target, transcription, remove_last_eot=True)
    
    inputs_embeds_target, attention_mask_target, (audio_start_target, audio_end_target) = desta_mt.prepare_llm_input(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            audio_position=audio_position,
            audio_features=audio_features
        )
    
  else:
    messages = [
                  *([{"role": "user", "content": user_prompt_target}] if user_prompt_target is not None else []),
                  {"role": "assistant", "content": prompt_target}
              ]
    inputs = desta_mt.process_text_without_audio(messages, remove_last_eot=True)
    inputs_embeds_target, attention_mask_target = desta_mt.prepare_llm_input_without_audio(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask
        )

  if position_target < 0:
    position_target = inputs_embeds_target.size(1) + position_target

  # handling patching from the end of the audio
  if patch_from_audio_end:
    original_position_source = position_source
    position_source = audio_end_source

  if patch_to_audio_end:
    original_position_target = position_target
    position_target = audio_end_target


  if layer_source < 0:
    layer_source = desta_mt.num_layers + layer_source
  
  if layer_target < 0:
    layer_target = desta_mt.num_layers + layer_target

  hs_patch_config = {
      layer_target: [(
          position_target,
          hs_cache_[layer_source][position_source],
      )]
  }

  if layer_source == layer_target == desta_mt.num_layers - 1:
    skip_final_ln = True
  else:
    skip_final_ln = False

  patch_hooks = desta_mt.set_hs_patch_hooks(
      desta_mt.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )
  
  if generation_mode:
    if temperature:
      output_toks = desta_mt.model.generate(
          inputs_embeds=inputs_embeds_target,
          attention_mask=attention_mask_target,
          max_new_tokens=max_gen_len,
          pad_token_id=desta_mt.model.config.eos_token_id,
          temperature=temperature,
          do_sample=True,
          top_k=0,
      )

    else:
      output_toks = desta_mt.model.generate(
          inputs_embeds=inputs_embeds_target,
          attention_mask=attention_mask_target,
          max_new_tokens=max_gen_len,
          pad_token_id=desta_mt.model.config.eos_token_id,
          do_sample=False
      )

    output = desta_mt.tokenizer.batch_decode(output_toks, skip_special_tokens=True)[0]
    if verbose:
      print(
          "generation with patching: ",
          [desta_mt.tokenizer.decode(x) for x in output_toks],
      )
  else:
    output = desta_mt.model(inputs_embeds=inputs_embeds_target, attention_mask=attention_mask_target, output_hidden_states=False)
  # remove patching hooks
  remove_hooks(patch_hooks)

  return output


def inspect_desta_loop_across_layers(
    desta_mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    generation_mode=False,
    max_gen_len=20,
    verbose=False,
    temperature=None,
    audio_source=None, # Path to the audio file (source of patching)
    audio_target=None, # Path to the audio file (target of patching)
    patch_from_audio_end=False, # whether to patch from the end of the audio position
    patch_to_audio_end=False, # whether to patch to the end of the audio position
    user_prompt_source=None, # User prompt for the source
    user_prompt_target=None, # User prompt for the target
    loop_layer_source=True,
    loop_layer_target=True,
    options=None, # list of possible options
    forward_add_coefficient=None, # coefficient for forward add patching
):
  """Inspection via patching."""
  if audio_source is not None:
    messages = [
                  {"role": "audio", "content": audio_source},
                  *([{"role": "user", "content": user_prompt_source}] if user_prompt_source is not None else []),
                  {"role": "assistant", "content": prompt_source}
              ]
    audio_source, input_features = desta_mt.load_audio(messages)
    transcription, audio_features = desta_mt.speech_perception.generate(input_features)
    inputs, audio_position = desta_mt.process_text(messages, audio_source, transcription, remove_last_eot=True)
    
    inputs_embeds_source, attention_mask_source, (audio_start_source, audio_end_source) = desta_mt.prepare_llm_input(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, 
            audio_position=audio_position,
            audio_features=audio_features
        )
    
  else:
    messages = [
                  *([{"role": "user", "content": user_prompt_source}] if user_prompt_source is not None else []),
                  {"role": "assistant", "content": prompt_source}
              ]
    inputs = desta_mt.process_text_without_audio(messages, remove_last_eot=True)
    inputs_embeds_source, attention_mask_source = desta_mt.prepare_llm_input_without_audio(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

  
  
  # first run the the model on prompt_patch and get all hidden states.

  hs_cache_ = []
  # We manually store intermediate states that the model API does not expose
  store_hooks = []
  if module == "mlp":

    def store_mlp_hook(module, input, output):
      hs_cache_.append(output[0])

    for layer in desta_mt.model.model.layers:
      store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
  elif module == "attn":

    def store_attn_hook(module, input, output):
      hs_cache_.append(output[0].squeeze())

    for layer in desta_mt.model.model.layers:
      store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))

  output = desta_mt.model(inputs_embeds=inputs_embeds_source, attention_mask=attention_mask_source, output_hidden_states=True)
  if module == "hs":
    hs_cache_ = [
        output["hidden_states"][layer + 1][0] for layer in range(desta_mt.num_layers)
    ]

  remove_hooks(store_hooks)
  
  # now do a second run on prompt, while patching
  # a specific hidden state from the first run.

  if audio_target is not None: # for logit lens case
    messages = [
                  {"role": "audio", "content": audio_target},
                  *([{"role": "user", "content": user_prompt_target}] if user_prompt_target is not None else []),
                  {"role": "assistant", "content": prompt_target}
              ]
    audio_target, input_features = desta_mt.load_audio(messages)
    transcription, audio_features = desta_mt.speech_perception.generate(input_features)
    inputs, audio_position = desta_mt.process_text(messages, audio_target, transcription, remove_last_eot=True)
    
    inputs_embeds_target, attention_mask_target, (audio_start_target, audio_end_target) = desta_mt.prepare_llm_input(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            audio_position=audio_position,
            audio_features=audio_features
        )
    
  else:
    messages = [
                  *([{"role": "user", "content": user_prompt_target}] if user_prompt_target is not None else []),
                  {"role": "assistant", "content": prompt_target}
              ]
    inputs = desta_mt.process_text_without_audio(messages, remove_last_eot=True)
    inputs_embeds_target, attention_mask_target = desta_mt.prepare_llm_input_without_audio(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask
        )

  if position_target < 0:
    position_target = inputs_embeds_target.size(1) + position_target

  # handling patching from the end of the audio
  if patch_from_audio_end:
    original_position_source = position_source
    position_source = audio_end_source

  if patch_to_audio_end:
    original_position_target = position_target
    position_target = audio_end_target

  if layer_source < 0:
    layer_source = desta_mt.num_layers + layer_source
  
  if layer_target < 0:
    layer_target = desta_mt.num_layers + layer_target

  outputs = {}

  source_layer_ranges = [layer_source] if not loop_layer_source else range(desta_mt.num_layers)
  target_layer_ranges = [layer_target] if not loop_layer_target else range(desta_mt.num_layers)
  
  for layer_source_idx in tqdm(source_layer_ranges, desc="Layer Source", leave=False):
    outputs[f'source_layer_{layer_source_idx}'] = {}
    
    for layer_target_idx in tqdm(target_layer_ranges, desc="Layer Target", leave=False):
      hs_patch_config = {
          layer_target_idx: [(
              position_target,
              hs_cache_[layer_source_idx][position_source],
          )]
      }

      if layer_source_idx == layer_target_idx == desta_mt.num_layers - 1:
        skip_final_ln = True
      else:
        skip_final_ln = False
      if forward_add_coefficient is not None:
        patch_hooks = desta_mt.set_hs_patch_hooks(
            desta_mt.model,
            hs_patch_config,
            module=module,
            patch_input=False,
            skip_final_ln=skip_final_ln,
            generation_mode=True,
            forward_add_coefficient=forward_add_coefficient
        )
      else:
        patch_hooks = desta_mt.set_hs_patch_hooks(
            desta_mt.model,
            hs_patch_config,
            module=module,
            patch_input=False,
            skip_final_ln=skip_final_ln,
            generation_mode=True        
        )

      if generation_mode:
        if temperature:
          output_toks = desta_mt.model.generate(
              inputs_embeds=inputs_embeds_target,
              attention_mask=attention_mask_target,
              max_new_tokens=max_gen_len,
              pad_token_id=desta_mt.model.config.eos_token_id,
              temperature=temperature,
              do_sample=True,
              top_k=0,
          )

        else:
          output_toks = desta_mt.model.generate(
              inputs_embeds=inputs_embeds_target,
              attention_mask=attention_mask_target,
              max_new_tokens=max_gen_len,
              pad_token_id=desta_mt.model.config.eos_token_id,
              do_sample=False
          )

        output = desta_mt.tokenizer.batch_decode(output_toks, skip_special_tokens=True)[0]
    
        outputs[f'source_layer_{layer_source_idx}'][f'target_layer_{layer_target_idx}'] = output
        
      else:
        output = desta_mt.model(inputs_embeds=inputs_embeds_target, attention_mask=attention_mask_target, output_hidden_states=False)
        
        # get options tokens
        first_token_dict = {}
        for o in options:
            tokens = desta_mt.tokenizer.tokenize(o)
            ids = desta_mt.tokenizer.convert_tokens_to_ids(tokens)
            first_token_dict[o] = ids[0]
        
        prob = {}
        for o in options:
            score = torch.softmax(output.logits[0, position_target, :], dim=0)[first_token_dict[o]]
            prob[o] = score.item()
        
        outputs[f'source_layer_{layer_source_idx}'][f'target_layer_{layer_target_idx}'] = prob
      
      remove_hooks(patch_hooks)

  return outputs







def evaluate_patch_next_token_prediction(
    mt,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    position_prediction=-1,
    transform=None,
):
  """Evaluate next token prediction."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  # adjust position_target to be absolute rather than relative
  inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
  if position_target < 0:
    position_target = len(inp_target["input_ids"][0]) + position_target

  # first run the the model on without patching and get the results.
  inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
  output_orig = mt.model(**inp_source, output_hidden_states=True)
  dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
  _, answer_t_orig = torch.max(dist_orig, dim=0)
  hidden_rep = output_orig["hidden_states"][layer_source + 1][0][
      position_source
  ]
  if transform is not None:
    hidden_rep = transform(hidden_rep)

  # now do a second run on prompt, while patching the input hidden state.
  hs_patch_config = {layer_target: [(position_target, hidden_rep)]}
  if layer_source == layer_target == mt.num_layers - 1:
    skip_final_ln = True
  else:
    skip_final_ln = False
  patch_hooks = mt.set_hs_patch_hooks(
      mt.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )
  output = mt.model(**inp_target)
  dist = torch.softmax(output.logits[0, position_prediction, :], dim=0)
  _, answer_t = torch.max(dist, dim=0)

  # remove patching hooks
  remove_hooks(patch_hooks)

  prec_1 = (answer_t == answer_t_orig).detach().cpu().item()
  surprisal = -torch.log(dist_orig[answer_t]).detach().cpu().numpy()

  return prec_1, surprisal


def evaluate_patch_next_token_prediction_x_model(
    mt_1,
    mt_2,
    prompt_source,
    prompt_target,
    layer_source,
    layer_target,
    position_source,
    position_target,
    module="hs",
    position_prediction=-1,
    transform=None,
):
  """evaluate next token prediction across models."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  # adjust position_target to be absolute rather than relative
  inp_target = make_inputs(mt_2.tokenizer, [prompt_target], device=mt_2.device)
  if position_target < 0:
    position_target = len(inp_target["input_ids"][0]) + position_target

  # first run the the model on without patching and get the results.
  inp_source = make_inputs(mt_1.tokenizer, [prompt_source], device=mt_1.device)
  output_orig = mt_1.model(**inp_source, output_hidden_states=True)
  dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
  _, answer_t_orig = torch.max(dist_orig, dim=0)
  hidden_rep = output_orig["hidden_states"][layer_source + 1][0][
      position_source
  ]
  if transform is not None:
    hidden_rep = transform(hidden_rep)

  # now do a second run on prompt, while patching the input hidden state.
  hs_patch_config = {layer_target: [(position_target, hidden_rep)]}
  skip_final_ln = False
  patch_hooks = mt_2.set_hs_patch_hooks(
      mt_2.model,
      hs_patch_config,
      module=module,
      patch_input=False,
      skip_final_ln=skip_final_ln,
      generation_mode=True,
  )
  output = mt_2.model(**inp_target)
  dist = torch.softmax(output.logits[0, position_prediction, :], dim=0)
  _, answer_t = torch.max(dist, dim=0)

  # remove patching hooks
  remove_hooks(patch_hooks)

  prec_1 = answer_t.detach().cpu().item() == answer_t_orig.detach().cpu().item()
  surprisal = -torch.log(dist_orig[answer_t]).detach().cpu().numpy()

  return prec_1, surprisal


# Adding support for batched patching. More than 10x speedup
# Currently only supporting GPT-J
def set_hs_patch_hooks_gptj_batch(
    model,
    hs_patch_config,
    module="hs",
    patch_input=False,
    generation_mode=False,
):
  """GPTJ patch hooks - supporting batch."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, inp):
      # (batch, sequence, hidden_state)
      idx_, position_, hs_ = (
          position_hs["batch_idx"],
          position_hs["position_target"],
          position_hs["hidden_rep"],
      )
      input_len = len(inp[0][idx_])
      if generation_mode and input_len == 1:
        return
      inp[0][idx_][position_] = hs_

    def post_hook(module, inp, output):
      idx_, position_, hs_ = (
          position_hs["batch_idx"],
          position_hs["position_target"],
          position_hs["hidden_rep"],
      )
      if "skip_ln" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[idx_])
        if generation_mode and output_len == 1:
          return
        output[idx_][position_] = hs_
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][idx_])
        if generation_mode and output_len == 1:
          return
        output[0][idx_][position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []
  for item in hs_patch_config:
    i = item["layer_target"]
    skip_final_ln = item["skip_final_ln"]
    if patch_input:
      hooks.append(
          model.transformer.h[i].register_forward_pre_hook(
              patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
          )
      )
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already
      # applied (assuming that the representation for patching was obtained by
      # setting output_hidden_representations to True).
      if skip_final_ln and i == len(model.transformer.h) - 1:
        hooks.append(
            model.transformer.ln_f.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln",
                    item,
                    patch_input,
                    generation_mode,
                )
            )
        )
      else:
        hooks.append(
            model.transformer.h[i].register_forward_hook(
                patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
            )
        )

  return hooks


def set_hs_patch_hooks_llama_batch(
    model,
    hs_patch_config,
    module="hs",
    patch_input=False,
    generation_mode=False,
):
  """LLAMA patch hooks - supporting batch."""
  # when using mode.generate() the hidden states in the input are cached after
  # the first inference pass, and in the next steps the input/output are of
  # size 1. In these cases we don't need to patch anymore the previous hidden
  # states from the initial input, because they are cached, but we do need to
  # handle these cases in this call because this hook wraps the generation call.
  #
  # NOTE: To use generation mode, we must patch a position that is not the
  # first one. This is because in this case we don't know during generation if
  # we are handling the initial input or a future step and thus don't know if
  # a patching is needed or not.

  # if generation_mode:
  #     for i in hs_patch_config:
  #         for position_, _ in hs_patch_config[i]:
  #             assert position_ > 0

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def patch_hs(name, position_hs, patch_input, generation_mode):
    def pre_hook(module, inp):
      # inp[0]: (batch, sequence, hidden_state)
      idx_, position_, hs_ = (
          position_hs["batch_idx"],
          position_hs["position_target"],
          position_hs["hidden_rep"],
      )
      input_len = len(inp[0][idx_])
      if generation_mode and input_len == 1:
        return
      inp[0][idx_][position_] = hs_

    def post_hook(module, inp, output):
      idx_, position_, hs_ = (
          position_hs["batch_idx"],
          position_hs["position_target"],
          position_hs["hidden_rep"],
      )
      if "skip_ln" in name:
        # output: (batch, sequence, hidden_state)
        output_len = len(output[idx_])
        if generation_mode and output_len == 1:
          return
        output[idx_][position_] = hs_
      else:
        # output[0]: (batch, sequence, hidden_state)
        output_len = len(output[0][idx_])
        if generation_mode and output_len == 1:
          return
        output[0][idx_][position_] = hs_

    if patch_input:
      return pre_hook
    else:
      return post_hook

  hooks = []

  for item in hs_patch_config:
    i = item["layer_target"]
    skip_final_ln = item["skip_final_ln"]
    if patch_input:
      hooks.append(
          model.model.layers[i].register_forward_pre_hook(
              patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
          )
      )
    else:
      # when patching a last-layer representation to the last layer of the same
      # model, the final layer norm is not needed because it was already applied
      # (assuming that the representation for patching was obtained by setting
      # output_hidden_representations to True).
      if skip_final_ln and i == len(model.model.layers) - 1:
        hooks.append(
            model.model.norm.register_forward_hook(
                patch_hs(
                    f"patch_hs_{i}_skip_ln", item, patch_input, generation_mode
                )
            )
        )
      else:
        hooks.append(
            model.model.layers[i].register_forward_hook(
                patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
            )
        )

  return hooks


def evaluate_patch_next_token_prediction_batch(
    mt, df, batch_size=256, transform=None, module="hs"
):
  """Evaluate next token prediction with batch support."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  prec_1 = np.zeros(0)
  surprisal = np.zeros(0)
  next_token = np.zeros(0)
  #     generations = []

  def _evaluat_single_batch(batch_df):
    batch_size = len(batch_df)
    prompt_source_batch = np.array(batch_df["prompt_source"])
    prompt_target_batch = np.array(batch_df["prompt_target"])
    layer_source_batch = np.array(batch_df["layer_source"])
    layer_target_batch = np.array(batch_df["layer_target"])
    position_source_batch = np.array(batch_df["position_source"])
    position_target_batch = np.array(batch_df["position_target"])
    position_prediction_batch = np.ones_like(position_target_batch) * -1
    #         max_gen_len = np.array(batch_df["max_gen_len"])

    # adjust position_target to be absolute rather than relative
    inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
    for i in range(batch_size):
      if position_target_batch[i] < 0:
        position_target_batch[i] += len(inp_target["input_ids"][i])

    # first run the the model on without patching and get the results.
    inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
    output_orig = mt.model(**inp_source, output_hidden_states=True)
    dist_orig = torch.softmax(
        output_orig.logits[
            np.array(range(batch_size)), position_source_batch, :
        ],
        dim=-1,
    )
    _, answer_t_orig = torch.max(dist_orig, dim=-1)
    # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
    hidden_rep = [
        output_orig.hidden_states[layer_source_batch[i] + 1][i][
            position_source_batch[i]
        ]
        for i in range(batch_size)
    ]
    if transform is not None:
      for i in range(batch_size):
        hidden_rep[i] = transform(hidden_rep[i])

    # now do a second run on prompt, while patching the input hidden state.
    hs_patch_config = [
        {
            "batch_idx": i,
            "layer_target": layer_target_batch[i],
            "position_target": position_target_batch[i],
            "hidden_rep": hidden_rep[i],
            "skip_final_ln": (
                layer_source_batch[i]
                == layer_target_batch[i]
                == mt.num_layers - 1
            ),
        }
        for i in range(batch_size)
    ]
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        generation_mode=False,
    )

    output = mt.model(**inp_target)

    # # NOTE: inputs are left padded,
    # # and sequence length is the same across batch
    # # to support generations of variable lengths,
    # # first generate with maximum number of tokens needed in the batch
    # seq_len = len(inp_target["input_ids"][0])
    # output_toks = mt.model.generate(
    #     inp_target["input_ids"],
    #     max_length=seq_len + max(max_gen_len),
    #     pad_token_id=mt.model.generation_config.eos_token_id,
    # )[:, seq_len:]

    # # then, we select only the subset of tokens that we need
    # generations = [
    #     mt.tokenizer.decode(output_toks[i][: max_gen_len[i]])
    #     for i in range(batch_size)
    # ]

    dist = torch.softmax(
        output.logits[
            np.array(range(batch_size)), position_prediction_batch, :
        ],
        dim=-1,
    )
    _, answer_t = torch.max(dist, dim=-1)
    next_token = [mt.tokenizer.decode(tok) for tok in answer_t]

    # remove patching hooks
    remove_hooks(patch_hooks)

    prec_1 = (answer_t == answer_t_orig).detach().cpu().numpy()
    surprisal = (
        -torch.log(dist_orig[np.array(range(batch_size)), answer_t])
        .detach()
        .cpu()
        .numpy()
    )

    return prec_1, surprisal, next_token

  for i in tqdm.tqdm(range(len(df) // batch_size)):
    cur_df = df.iloc[batch_size * i : batch_size * (i + 1)]
    batch_prec_1, batch_surprisal, batch_next_token = _evaluat_single_batch(
        cur_df
    )
    prec_1 = np.concatenate((prec_1, batch_prec_1))
    surprisal = np.concatenate((surprisal, batch_surprisal))
    next_token = np.concatenate((next_token, batch_next_token))

  return prec_1, surprisal, next_token


def inspect_batch(mt, df, batch_size=256, transform=None, module="hs"):
  """Inspects batch: source/target layer/position could differ within batch."""
  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  generations = []

  def _inspect_single_batch(batch_df):
    batch_size = len(batch_df)
    prompt_source_batch = np.array(batch_df["prompt_source"])
    prompt_target_batch = np.array(batch_df["prompt_target"])
    layer_source_batch = np.array(batch_df["layer_source"])
    layer_target_batch = np.array(batch_df["layer_target"])
    position_source_batch = np.array(batch_df["position_source"])
    position_target_batch = np.array(batch_df["position_target"])
    max_gen_len = np.array(batch_df["max_gen_len"])

    # adjust position_target to be absolute rather than relative
    inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
    for i in range(batch_size):
      if position_target_batch[i] < 0:
        position_target_batch[i] += len(inp_target["input_ids"][i])

    # first run the the model on without patching and get the results.
    inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
    output_orig = mt.model(**inp_source, output_hidden_states=True)

    # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
    hidden_rep = [
        output_orig.hidden_states[layer_source_batch[i] + 1][i][
            position_source_batch[i]
        ]
        for i in range(batch_size)
    ]
    if transform is not None:
      for i in range(batch_size):
        hidden_rep[i] = transform(hidden_rep[i])

    # now do a second run on prompt, while patching the input hidden state.
    hs_patch_config = [
        {
            "batch_idx": i,
            "layer_target": layer_target_batch[i],
            "position_target": position_target_batch[i],
            "hidden_rep": hidden_rep[i],
            "skip_final_ln": (
                layer_source_batch[i]
                == layer_target_batch[i]
                == mt.num_layers - 1
            ),
        }
        for i in range(batch_size)
    ]
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        generation_mode=True,
    )

    # NOTE: inputs are left padded,
    # and sequence length is the same across batch
    # to support generations of variable lengths,
    # first generate with maximum number of tokens needed in the batch
    seq_len = len(inp_target["input_ids"][0])
    output_toks = mt.model.generate(
        inp_target["input_ids"],
        max_length=seq_len + max(max_gen_len),
        pad_token_id=mt.model.generation_config.eos_token_id,
    )[:, seq_len:]

    # then, we select only the subset of tokens that we need
    generations = [
        mt.tokenizer.decode(output_toks[i][: max_gen_len[i]])
        for i in range(batch_size)
    ]

    # remove patching hooks
    remove_hooks(patch_hooks)

    return generations

  for i in tqdm.tqdm(range(1 + len(df) // batch_size)):
    cur_df = df.iloc[batch_size * i : batch_size * (i + 1)]
    batch_generations = _inspect_single_batch(cur_df)
    generations.extend(batch_generations)

  return generations


def evaluate_attriburte_exraction_batch(
    mt,
    df,
    batch_size=256,
    max_gen_len=10,
    transform=None,
    is_icl=True,
    module="hs",
):
  """Evaluates attribute extraction with batch support."""
  # We don't know the exact token position of the
  # attribute, as it is not necessarily the next token. So, precision and
  # surprisal may not apply directly.

  if module != "hs":
    raise ValueError("Module %s not yet supported", module)

  def _evaluate_attriburte_exraction_single_batch(batch_df):
    batch_size = len(batch_df)
    prompt_source_batch = np.array(batch_df["prompt_source"])
    prompt_target_batch = np.array(batch_df["prompt_target"])
    layer_source_batch = np.array(batch_df["layer_source"])
    layer_target_batch = np.array(batch_df["layer_target"])
    position_source_batch = np.array(batch_df["position_source"])
    position_target_batch = np.array(batch_df["position_target"])

    object_batch = np.array(batch_df["object"])

    # Adjust position_target to be absolute rather than relative
    inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
    for i in range(batch_size):
      if position_target_batch[i] < 0:
        position_target_batch[i] += len(inp_target["input_ids"][i])

    # Step 1: run model on source prompt without patching and get the hidden
    # representations.
    inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
    output_orig = mt.model(**inp_source, output_hidden_states=True)

    # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
    #         hidden_rep = []
    #         for i in range(batch_size):
    #             hidden_rep.append(output_orig.hidden_states[layer_source_batch[i] + 1][i][position_source_batch[i]])
    hidden_rep = [
        output_orig.hidden_states[layer_source_batch[i] + 1][i][
            position_source_batch[i]
        ]
        for i in range(batch_size)
    ]
    if transform is not None:
      for i in range(batch_size):
        hidden_rep[i] = transform(hidden_rep[i])

    # Step 2: Do second run on target prompt, while patching the input
    # hidden state.
    hs_patch_config = [
        {
            "batch_idx": i,
            "layer_target": layer_target_batch[i],
            "position_target": position_target_batch[i],
            "hidden_rep": hidden_rep[i],
            "skip_final_ln": (
                layer_source_batch[i]
                == layer_target_batch[i]
                == mt.num_layers - 1
            ),
        }
        for i in range(batch_size)
    ]
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        generation_mode=True,
    )

    # Note that inputs are left padded,
    # and sequence length is the same across batch
    seq_len = len(inp_target["input_ids"][0])
    output_toks = mt.model.generate(
        inp_target["input_ids"],
        max_length=seq_len + max_gen_len,
        pad_token_id=mt.model.generation_config.eos_token_id,
    )[:, seq_len:]
    generations_patched = decode_tokens(mt.tokenizer, output_toks)
    if is_icl:
      prefix = batch_df["prefix"].iloc[0]

      def _crop_by_prefix(generations, prefix):
        concatenated_str = " ".join(generations)
        _pos = concatenated_str.find(prefix)
        return concatenated_str[:_pos]

      generations_patched_postprocessed = np.array([
          _crop_by_prefix(generations_patched[i], prefix)
          for i in range(batch_size)
      ])
    else:
      generations_patched_postprocessed = np.array(
          [" ".join(generations_patched[i]) for i in range(batch_size)]
      )

    is_correct_patched = np.array([
        object_batch[i].replace(" ", "")
        in generations_patched_postprocessed[i].replace(" ", "")
        for i in range(batch_size)
    ])

    # remove patching hooks
    remove_hooks(patch_hooks)

    cpu_hidden_rep = np.array(
        [hidden_rep[i].detach().cpu().numpy() for i in range(batch_size)]
    )

    results = {
        "generations_patched": generations_patched,
        "generations_patched_postprocessed": generations_patched_postprocessed,
        "is_correct_patched": is_correct_patched,
        "hidden_rep": cpu_hidden_rep,
    }

    return results

  results = {}
  n_batches = len(df) // batch_size
  if len(df) % batch_size != 0:
    n_batches += 1
  for i in tqdm(range(len(df) // batch_size)):
    cur_df = df.iloc[batch_size * i : batch_size * (i + 1)]
    batch_results = _evaluate_attriburte_exraction_single_batch(cur_df)
    for key, value in batch_results.items():
      if key in results:
        results[key] = np.concatenate((results[key], value))
      else:
        results[key] = value

  return results
