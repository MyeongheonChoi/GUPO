import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers import get_cosine_schedule_with_warmup
from omegaconf import DictConfig

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    rank0_print,
    get_local_dir,
)
import numpy as np
import pandas as pd
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

def stabilized_log1pexp(x):
    """
    Compute log(1 + exp(x)) in a numerically stable way."""
    mask = x > 0
    risky_x = torch.where(mask, x, torch.zeros_like(x))
    safe_x = torch.where(mask, torch.zeros_like(x), x)
    return torch.where(mask, risky_x + torch.log1p(torch.exp(-risky_x)), torch.log1p(torch.exp(safe_x)))


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.autocast_dtype = torch.float32
        self.autocast_enabled = False
        
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
                self.autocast_enabled = True
                rank0_print("✅ bfloat16 is supported. Using bfloat16 mixed precision (autocast).")
            else:
                rank0_print("⚠️ bfloat16 is NOT supported. Running in full float32 for stability.")
        else:
            rank0_print("CUDA is NOT available. Running on CPU (if configured).")

        self.data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        self.eval_iterator = get_batch_iterator(
            **self.data_iterator_kwargs, 
            split='test', 
            n_examples=config.n_eval_examples, 
            batch_size=config.eval_batch_size,
            shuffle=False,
            silent=rank != 0, 
            cache_dir=get_local_dir(config.local_dirs)
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
        
        if config.loss.name == 'dpo':
            self._loss_fn = dpo_loss
            

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        policy_output = self.policy.generate(
            batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        if self.config.loss.name == 'dpo':
            reference_output = self.reference_model.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name == 'dpo':
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes..
        """
        concatenated_batch = concatenated_inputs(batch)
        
        with torch.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.autocast_enabled):
            all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
            all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
            chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
            rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
            return chosen_logps, rejected_logps


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'dpo':
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            # losses, chosen_rewards, rejected_rewards = dpo_loss(
            #     policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=loss_config.beta, reference_free=loss_config.reference_free)
            losses, chosen_rewards, rejected_rewards = self._loss_fn(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=loss_config.beta, reference_free=loss_config.reference_free)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)

        rank0_print("Calculating steps per epoch...")
        temp_epoch_iterator = get_batch_iterator(
            **self.data_iterator_kwargs,
            split = 'train',
            n_epochs = 1,
            n_examples = self.config.n_examples,
            batch_size = self.config.batch_size,
            silent = True,
            cache_dir = get_local_dir(self.config.local_dirs)
        )
        
        steps_per_epoch = len(list(temp_epoch_iterator))
        del temp_epoch_iterator
        
        if steps_per_epoch == 0:
            raise ValueError("The training dataset is empty. Please check the dataset configuration.")
        
        total_training_steps = steps_per_epoch * self.config.n_epochs
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = self.config.warmup_steps,
            num_training_steps = total_training_steps
        )
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name == 'dpo':
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for epoch in range(1, self.config.n_epochs + 1):
            rank0_print(f'===== Starting epoch {epoch}/{self.config.n_epochs} =====')
            
            epoch_train_iterator = get_batch_iterator(
                **self.data_iterator_kwargs,
                split = 'train',
                n_epochs = 1,
                n_examples = self.config.n_examples,
                batch_size = self.config.batch_size,
                shuffle = True,
                silent = self.rank != 0,
                cache_dir = get_local_dir(self.config.local_dirs)
            )
            
            for batch in epoch_train_iterator:
                #### BEGIN EVALUATION ####
                if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                    rank0_print(f'Running evaluation after {self.example_counter} train examples')
                    self.policy.eval()

                    all_eval_metrics = defaultdict(list)
                    if self.config.sample_during_eval:
                        all_policy_samples, all_reference_samples = [], []
                        policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                        if self.config.loss.name == 'dpo':
                            reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                    for eval_batch in (tqdm.tqdm(self.eval_batches) if self.rank == 0 else self.eval_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        with torch.no_grad():
                            _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                        for k, v in eval_metrics.items():
                            all_eval_metrics[k].extend(v)

                        if self.config.sample_during_eval:
                            policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                            all_policy_samples.extend(policy_samples)
                            all_reference_samples.extend(reference_samples)

                            for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                                policy_text_table.add_data(self.example_counter, prompt, sample)
                            if self.config.loss.name == 'dpo':
                                for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                    reference_text_table.add_data(self.example_counter, prompt, sample)

                    mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                    rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                    if self.config.sample_during_eval:                    
                        rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                        if self.config.loss.name == 'dpo':
                            rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                    if self.config.wandb.enabled and self.rank == 0:
                        wandb.log(mean_eval_metrics, step=self.example_counter)

                        if self.config.sample_during_eval:
                            wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                            if self.config.loss.name == 'dpo':
                                wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                    if self.example_counter > 0:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
                #### END EVALUATION ####

                #### BEGIN TRAINING ####
                self.policy.train()

                start_time = time.time()
                batch_metrics = defaultdict(list)
                for microbatch_idx in range(self.config.gradient_accumulation_steps):
                    global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                    local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                    loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                    (loss / self.config.gradient_accumulation_steps).backward()

                    for k, v in metrics.items():
                        batch_metrics[k].extend(v)

                grad_norm = self.clip_gradient()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                step_time = time.time() - start_time
                examples_per_second = self.config.batch_size / step_time
                batch_metrics['examples_per_second'].append(examples_per_second)
                batch_metrics['grad_norm'].append(grad_norm)
                batch_metrics['lr'].append(self.scheduler.get_last_lr()[0])

                self.batch_counter += 1
                self.example_counter += self.config.batch_size

                if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                    mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                    mean_train_metrics['counters/examples'] = self.example_counter
                    mean_train_metrics['counters/updates'] = self.batch_counter
                    rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                    if self.config.wandb.enabled and self.rank == 0:
                        wandb.log(mean_train_metrics, step=self.example_counter)

                    last_log = time.time()
                else:
                    rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
                    
            rank0_print(f'End of Epoch {epoch}. Saving checkpoint...')
            
            output_dir = os.path.join(self.run_dir, f'epoch-{epoch}_step-{self.example_counter}')
            self.save(output_dir, metrics=None)
            
            rank0_print(f'Checkpoint saved to {output_dir}')
                #### END TRAINING ####


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()
    
    def generate_beta_dataset(self):
        """
        Evaluates the MLP on the FULL evaluation dataset (not just the cached subset)
        and returns a DataFrame.
        """
        rank0_print(f'Generating Beta Dataset for the FULL test set...')
        
        self.policy.eval()

        full_eval_iterator = get_batch_iterator(
            **self.data_iterator_kwargs,
            split='test',
            shuffle=False,
            n_examples=None,          
            n_epochs=1,
            batch_size=self.config.eval_batch_size,
            silent=(self.rank != 0),
            cache_dir=get_local_dir(self.config.local_dirs)
        )

        results = []

        iterator_wrapper = tqdm.tqdm(full_eval_iterator, desc="Generating Full Beta Data") if self.rank == 0 else full_eval_iterator

        for eval_batch in iterator_wrapper:
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            
            with torch.no_grad():
                chosen_logps, rejected_logps = self.concatenated_forward(
                    self.policy, local_eval_batch
                )
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                    self.reference_model, local_eval_batch
                )
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            logits = pi_logratios - ref_logratios
            beta = self.config.loss.beta
            prob = F.sigmoid(beta * logits)
            
            chosen_rewards = beta * (chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = beta * (rejected_logps - reference_rejected_logps).detach()
            
            reward_margin = chosen_rewards - rejected_rewards

            if 'prompt' in eval_batch:
                prompts = eval_batch['prompt']
                chosens = eval_batch.get('chosen', eval_batch.get('chosen_response', ["N/A"] * len(prompts)))
                rejects = eval_batch.get('rejected', eval_batch.get('rejected_response', ["N/A"] * len(prompts)))
                
                if len(chosens) > 0 and not isinstance(chosens[0], str):
                     chosens = self.tokenizer.batch_decode(eval_batch['chosen_input_ids'], skip_special_tokens=True)
                if len(rejects) > 0 and not isinstance(rejects[0], str):
                     rejects = self.tokenizer.batch_decode(eval_batch['rejected_input_ids'], skip_special_tokens=True)
            else:
                prompts = self.tokenizer.batch_decode(eval_batch['prompt_input_ids'], skip_special_tokens=True)
                chosens = self.tokenizer.batch_decode(eval_batch['chosen_input_ids'], skip_special_tokens=True)
                rejects = self.tokenizer.batch_decode(eval_batch['rejected_input_ids'], skip_special_tokens=True)

            batch_prob = prob.float().cpu().tolist()
            batch_reward_margin = reward_margin.float().cpu().tolist()

            for p, c, r, prob, rm in zip(prompts, chosens, rejects, batch_prob, batch_reward_margin):
                
                if c.startswith(p):
                    c = c[len(p):]
                
                if r.startswith(p):
                    r = r[len(p):]
                
                c = c.strip()
                r = r.strip()
                
                results.append({
                    "prompt": p,
                    "chosen_response": c,
                    "rejected_response": r,
                    "probability": prob,
                    "reward_margin": rm,
                })
       
        df = pd.DataFrame(results)
        rank0_print(f"Generated dataset with {len(df)} samples.")
        return df

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""
        is_peft_model = False
        try:
            from peft import PeftModel
            if isinstance(self.policy, PeftModel):
                is_peft_model = True
        except Exception as e:
            rank0_print('PEFT not installed or policy is not a PEFT model; skipping adapter save.', e)

        if is_peft_model:
            adapter_dir = os.path.join(output_dir if output_dir is not None else os.path.join(self.run_dir, f'LATEST'), 'adapter')
            os.makedirs(adapter_dir, exist_ok=True)
            rank0_print(f'writing checkpoint to {adapter_dir}...')
            self.policy.save_pretrained(adapter_dir)
            rank0_print(f'Saved PEFT adapter to {adapter_dir}')
        else:
            policy_state_dict = self.policy.state_dict()
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
            del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        del scheduler_state_dict

        print('Done.')
