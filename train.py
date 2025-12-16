import torch
from datetime import datetime

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed
import os
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig

from trainers.dpo_trainers import BasicTrainer
from trainers.gupo_trainers import GUPOTrainer
from trainers.simpo_trainers import SimPOTrainer
from trainers.alphadpo_trainers import AlphaDPOTrainer

import wandb
import json
import socket
from typing import Optional, Set


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )
    if config.loss.name == 'dpo':
        TrainerClass = BasicTrainer
    elif config.loss.name == 'gupo':
        TrainerClass = GUPOTrainer
    elif config.loss.name == 'simpo':
        TrainerClass = SimPOTrainer
    elif config.loss.name == 'alphadpo':
        TrainerClass = AlphaDPOTrainer
    elif config.loss.name == 'sft':
        TrainerClass = BasicTrainer
    else:
        raise ValueError(f'Unknown loss name: {config.loss.name}')
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(
        policy, 
        config, 
        config.seed, 
        config.local_run_dir, 
        reference_model=reference_model, 
        rank=rank, 
        world_size=world_size
    )

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    print(OmegaConf.to_yaml(config))
    # now = datetime.now()
    # time_string = now.strftime("%M %H %S")
    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'}
    if config.model.policy_quantization == '8bit':
        print('using 8-bit quantization for policy model')
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model_kwargs['quantization_config'] = bnb_config
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, 
        cache_dir=get_local_dir(config.local_dirs), 
        low_cpu_mem_usage=True, 
        dtype=policy_dtype, 
        **model_kwargs
    )
    
    if config.lora.enabled:
        print('applying LoRA adapters')
        # if getattr(policy, 'is_loaded_in_8bit', False) or getattr(policy, 'is_loaded_in_4bit', False):
        #     policy = prepare_model_for_kbit_training(policy)
            
        lora_config = LoraConfig(
            r=config.lora.r,
            target_modules=list(config.lora.target_modules),
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        policy = get_peft_model(policy, lora_config)
        policy.print_trainable_parameters()
    
    disable_dropout(policy)

    if config.loss.name == 'dpo' or config.loss.name == 'gupo' or config.loss.name == 'alphadpo':
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, 
            cache_dir=get_local_dir(config.local_dirs), 
            low_cpu_mem_usage=True, 
            dtype=reference_model_dtype, 
            **model_kwargs
        )
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name == 'dpo':
            reference_model.load_state_dict(state_dict['state'])
        if config.loss.name == 'gupo':
            reference_model.load_state_dict(state_dict['state'])
        if config.loss.name == 'alphadpo':
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')
    \
    worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()