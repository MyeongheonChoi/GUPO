import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from transformers import get_cosine_schedule_with_warmup
from omegaconf import DictConfig

import torch.distributed as dist

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
import math
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

def gupo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             beta_chosen: torch.FloatTensor,
             beta_rejected: torch.FloatTensor,
             rho: float = 0) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the GUPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        beta_chosen: Per-example beta values for the chosen responses, predicted by a learned MLP. Shape: (batch_size,)
        beta_rejected: Per-example beta values for the rejected responses, predicted by a learned MLP. Shape: (batch_size,)
        rho: Correlation coefficient between the chosen and rejected responses. Default is 0 (independent).

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards, delta_V, mu, std, z).
        The losses tensor contains the GUPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        the delta_V, mu, std, z tensors contain intermediate values used in the loss computation.
    """
    # print(policy_chosen_logps)
    # print(policy_rejected_logps)
    # print(reference_chosen_logps)
    # print(reference_rejected_logps)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    gamma = 0.5772156649

    logits = pi_logratios - ref_logratios
    
    delta_V = beta * logits
    # delta_V = 5 * torch.tanh(logits / 5) # scale to (-5, 5)
    mu = gamma * (beta_rejected - beta_chosen)
    std_squared = (math.pi**2 / 6) * (beta_chosen**2 + beta_rejected**2 - 2 * rho * beta_chosen * beta_rejected)
    std = torch.sqrt(std_squared)
    # std = torch.sqrt(torch.clamp(std_squared, min=1e-8))
    z = (delta_V - mu) / std
    losses = - torch.special.log_ndtr(z)
    
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    # print(losses.shape)

    return losses, chosen_rewards, rejected_rewards, delta_V, mu, std, z


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


class GUPOTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting GUPO training.
           
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
            sft_mode=False,
        )

        self.policy = policy
        self.reference_model = reference_model
        self.mlp = nn.Sequential(
            nn.Linear(self.policy.config.hidden_size, self.policy.config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.policy.config.hidden_size // 4, 1),
            nn.Softplus()
            ).to(self.policy.device)

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


    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing GUPO training) for the given batch of inputs."""

        policy_output = self.policy.generate(
            batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        reference_output = self.reference_model.generate(
            batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
        reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded
    

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], compute_beta: bool) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes.
        """
        concatenated_batch = concatenated_inputs(batch)

        with torch.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.autocast_enabled):
            outputs = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], output_hidden_states=True)
            
            all_logits = outputs.logits.to(torch.float32)        
            all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
            chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
            rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
            
            if compute_beta:
                all_hidden_states = outputs.hidden_states[-1] # (batch_size * 2, seq_len, hidden_size)
                labels = concatenated_batch['concatenated_labels'][:, 1:].clone()
                loss_mask = (labels != -100)
                hidden_states_for_logps = all_hidden_states[:, :-1, :]
                # hidden state mean pooling
                masked_hidden_states = hidden_states_for_logps * loss_mask.unsqueeze(-1)
                sum_hidden_states = masked_hidden_states.sum(dim=1)  # (batch_size * 2, hidden_size)
                num_tokens = loss_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)  # (batch_size * 2, 1)
                mean_pooled_embeddings = sum_hidden_states / num_tokens  # (batch_size * 2, hidden_size)
                
                # print(mean_pooled_embeddings)
                # all_betas_raw = self.mlp(mean_pooled_embeddings.detach()).squeeze(-1)  # (batch_size * 2,)
                all_betas = self.mlp(mean_pooled_embeddings.detach()).squeeze(-1)  # (batch_size * 2,)
                # all_betas = all_betas_raw * 9.9 + 0.1  # scale to (0.1, 10.0)
                chosen_betas = all_betas[:batch['chosen_input_ids'].shape[0]]
                rejected_betas = all_betas[batch['chosen_input_ids'].shape[0]:]

                return chosen_logps.to(torch.float32), rejected_logps.to(torch.float32), chosen_betas.to(torch.float32), rejected_betas.to(torch.float32)
            
            else:
                return chosen_logps.to(torch.float32), rejected_logps.to(torch.float32), None, None


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the GUPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        policy_chosen_logps, policy_rejected_logps, beta_chosen, beta_rejected = self.concatenated_forward(self.policy, batch, compute_beta=True)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.reference_model, batch, compute_beta=False)

        losses, chosen_rewards, rejected_rewards, delta_V, mu, std, z = gupo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=loss_config.beta_dpo, beta_chosen=beta_chosen, beta_rejected=beta_rejected, rho=loss_config.rho)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
        rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
        reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

        metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
        
        beta_chosen = all_gather_if_needed(beta_chosen.detach(), self.rank, self.world_size)
        metrics[f'mlp_betas_{train_test}/chosen'] = beta_chosen.cpu().numpy().tolist()
        beta_rejected = all_gather_if_needed(beta_rejected.detach(), self.rank, self.world_size)
        metrics[f'mlp_betas_{train_test}/rejected'] = beta_rejected.cpu().numpy().tolist()

        delta_V = all_gather_if_needed(delta_V.detach(), self.rank, self.world_size)
        metrics[f'gupo_delta_V_{train_test}'] = delta_V.cpu().numpy().tolist()
        mu = all_gather_if_needed(mu.detach(), self.rank, self.world_size)
        metrics[f'gupo_mu_{train_test}'] = mu.cpu().numpy().tolist()
        std = all_gather_if_needed(std.detach(), self.rank, self.world_size)
        metrics[f'gupo_std_{train_test}'] = std.cpu().numpy().tolist()
        z = all_gather_if_needed(z.detach(), self.rank, self.world_size)
        metrics[f'gupo_z_{train_test}'] = z.cpu().numpy().tolist()
        

        policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
        
        main_loss = losses.mean()
        reg_lambda = loss_config.mlp_reg_lambda if train else 0.0
        
        total_loss = main_loss
        if reg_lambda > 0 and train:
            beta_reg_loss = ((beta_chosen - 1)**2) + ((beta_rejected - 1)**2)
            metrics[f'loss/{train_test}_mlp_reg'] = all_gather_if_needed(beta_reg_loss.detach(), self.rank, self.world_size).cpu().numpy().tolist()
            total_loss = total_loss + reg_lambda * beta_reg_loss.mean()

        return total_loss, metrics

    def perform_optimization_step(self, epoch: int, grad_norm: float, grad_norm_mlp: float) -> Dict:
        """
        Helper function to decide which optimizers to step based on the learning straetegy and current epoch.
        Returns a dictionary of metrics to be logged.
        """
        strategy = self.config.loss.learning_strategy
        metrics = {}

        step_policy = False
        step_mlp = False

        if strategy == 'joint':
            step_policy = True
            step_mlp = True
        elif strategy == 'alternating':
            if epoch % 2 == 1:
                step_mlp = True
            else:
                step_policy = True
        elif strategy == 'phased':
            if epoch == 1:
                step_mlp = True
            else:
                step_policy = True
                step_mlp = True
        else:
            raise ValueError(f'Unknown learning strategy: {strategy}')
        
        if step_policy:
            self.optimizer.step()
            self.scheduler.step()
            metrics['grad_norm'] = grad_norm
            metrics['lr'] = self.scheduler.get_last_lr()[0]
        else:
            self.optimizer.zero_grad()
            metrics['grad_norm'] = 0.0
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        if step_mlp:
            self.optimizer_mlp.step()
            self.scheduler_mlp.step()
            metrics['grad_norm_mlp'] = grad_norm_mlp
            metrics['lr_mlp'] = self.scheduler_mlp.get_last_lr()[0]
        else:
            self.optimizer_mlp.zero_grad()
            metrics['grad_norm_mlp'] = 0.0
            metrics['lr_mlp'] = self.optimizer_mlp.param_groups[0]['lr']

        self.optimizer.zero_grad()
        self.optimizer_mlp.zero_grad()

        return metrics


    def train(self):
        """Begin GUPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer for policy')
        rank0_print(f'Using {self.config.optimizer_mlp} optimizer for mlp')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.loss.lr)
        self.optimizer_mlp = getattr(torch.optim, self.config.optimizer_mlp)(self.mlp.parameters(), lr=self.config.loss.lr_mlp, weight_decay=0.01)
        
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

        strategy = self.config.loss.learning_strategy
        rank0_print(f"Using '{strategy}' learning strategy")

        total_policy_steps = 0
        total_mlp_steps = 0

        if strategy == 'joint':
            total_policy_steps = total_training_steps
            total_mlp_steps = total_training_steps
        elif strategy == 'alternating':
            policy_epochs = self.config.n_epochs // 2
            mlp_epochs = self.config.n_epochs - policy_epochs
            total_policy_steps = steps_per_epoch * policy_epochs
            total_mlp_steps = steps_per_epoch * mlp_epochs
        elif strategy == 'phased':
            total_mlp_steps = steps_per_epoch
            total_policy_steps = steps_per_epoch * (self.config.n_epochs - 1)
        
        if total_policy_steps <= 0:
            rank0_print("Warning: Total training steps for policy is less than or equal to zero. Adjusting warmup steps accordingly.")
            total_policy_steps = 1
        if total_mlp_steps <= 0:
            rank0_print("Warning: Total training steps for mlp is less than or equal to zero. Adjusting warmup steps accordingly.")
            total_mlp_steps = 1

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = self.config.warmup_steps,
            num_training_steps = total_policy_steps
        )
        self.scheduler_mlp = get_cosine_schedule_with_warmup(
            self.optimizer_mlp,
            num_warmup_steps = 0,
            num_training_steps = total_mlp_steps
        )
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

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
            
            is_pahsed_warmup = (strategy == 'phased' and epoch == 1)
            is_alternating_mlp = (strategy == 'alternating' and epoch % 2 == 1)

            skip_eval_this_epoch = is_pahsed_warmup or is_alternating_mlp
            if skip_eval_this_epoch:
                rank0_print(f'Epoch {epoch} is an MLP-only training phase. Skipping evaluation.')

            for batch in epoch_train_iterator:
                
                #### BEGIN EVALUATION ####
                if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval) and not skip_eval_this_epoch:
                    rank0_print(f'Running evaluation after {self.example_counter} train examples')
                    self.policy.eval()
                    self.mlp.eval()

                    all_eval_metrics = defaultdict(list)
                    if self.config.sample_during_eval:
                        all_policy_samples, all_reference_samples = [], []
                        policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                        if self.config.loss.name == 'gupo':
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
                            if self.config.loss.name == 'gupo':
                                for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                    reference_text_table.add_data(self.example_counter, prompt, sample)

                    mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                    rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                    if self.config.sample_during_eval:                    
                        rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                        if self.config.loss.name == 'gupo':
                            rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                    if self.config.wandb.enabled and self.rank == 0:
                        wandb.log(mean_eval_metrics, step=self.example_counter)

                        if self.config.sample_during_eval:
                            wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                            if self.config.loss.name == 'gupo':
                                wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                    if self.example_counter > 0:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
                #### END EVALUATION ####

                #### BEGIN TRAINING ####
                self.policy.train()
                self.mlp.train()

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
                grad_norm_mlp = self.clip_gradient_mlp()

                step_metrics = self.perform_optimization_step(epoch, grad_norm, grad_norm_mlp)
                for k, v in step_metrics.items():
                    batch_metrics[k].append(v)

                step_time = time.time() - start_time
                examples_per_second = self.config.batch_size / step_time
                batch_metrics['examples_per_second'].append(examples_per_second)
                
                self.batch_counter += 1
                self.example_counter += self.config.batch_size

                if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                    mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                    mean_train_metrics['counters/examples'] = self.example_counter
                    mean_train_metrics['counters/updates'] = self.batch_counter
                    mean_train_metrics['counters/epoch'] = epoch
                    rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                    if self.config.wandb.enabled and self.rank == 0:
                        wandb.log(mean_train_metrics, step=self.example_counter)

                    last_log = time.time()
                else:
                    rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####

            rank0_print(f'End of Epoch {epoch}. Saving checkpoint...')
            
            # 에폭 기반의 폴더 이름 생성 (예: epoch-1_step-5000)
            output_dir = os.path.join(self.run_dir, f'epoch-{epoch}_step-{self.example_counter}')
            
            # 평가를 스킵했을 경우 metrics는 None으로 전달합니다.
            # (save 함수가 metrics=None을 처리하도록 되어 있으므로 안전합니다)
            self.save(output_dir, metrics=None)
            
            rank0_print(f'Checkpoint saved to {output_dir}')


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def clip_gradient_mlp(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), self.config.loss.max_grad_norm_mlp).item()

    def evaluate_mlp(self):
        """
        Runs a fast evaluation pass focused *only* on the get_batch_metrics
        to check loss, rewards, and MLP beta values, skipping slow generation.
        """        
        # 1. Set models to evaluation mode
        self.policy.eval()
        self.mlp.eval()

        all_eval_metrics = defaultdict(list)

        # 2. Iterate over eval batches
        for eval_batch in (tqdm.tqdm(self.eval_batches) if self.rank == 0 else self.eval_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            
            with torch.no_grad():
                # 3. Call get_batch_metrics (No generation)
                # This function already calculates 'mlp_betas_eval/chosen' etc.
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        # 4. Calculate mean metrics
        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        rank0_print(f'eval after: {formatted_dict(mean_eval_metrics)}')
            
        return mean_eval_metrics

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

        mlp_state_dict = self.mlp.state_dict()
        self.write_state_dict(self.example_counter, mlp_state_dict, metrics, 'mlp.pt', output_dir)
        del mlp_state_dict
        
        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        
        optimizer_mlp_state_dict = self.optimizer_mlp.state_dict()
        self.write_state_dict(self.example_counter, optimizer_mlp_state_dict, metrics, 'optimizer_mlp.pt', output_dir)
        del optimizer_mlp_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        del scheduler_state_dict

        scheduler_mlp_state_dict = self.scheduler_mlp.state_dict()
        self.write_state_dict(self.example_counter, scheduler_mlp_state_dict, metrics, 'scheduler_mlp.pt', output_dir)
        del scheduler_mlp_state_dict
        
        

        print('Done.')