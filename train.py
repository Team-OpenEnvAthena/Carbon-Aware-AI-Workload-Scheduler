"""
train.py — TRL + Unsloth GRPO training for CarbonSchedulerEnv

Run in Colab:
  !pip install unsloth trl transformers accelerate
  !python train.py

Or mount the env locally:
  python train.py --env_url http://localhost:7860 --steps 500
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
parser.add_argument("--env_url",    default=None,  help="Remote env URL. None = local.")
parser.add_argument("--steps",      type=int, default=300)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr",         type=float, default=5e-6)
parser.add_argument("--output_dir", default="./carbon_scheduler_model")
parser.add_argument("--curriculum", type=int, default=1, choices=[1,2,3],
                    help="Curriculum level: 1=easy, 2=medium, 3=hard")
parser.add_argument("--log_wandb",  action="store_true")
args = parser.parse_args()


# ── Imports ───────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import torch

if args.log_wandb:
    import wandb
    wandb.init(project="carbon-scheduler-rl", config=vars(args))


# ── Load model with Unsloth ───────────────────────────────────────────────────
print(f"Loading model: {args.model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = args.model,
    max_seq_length = 2048,
    load_in_4bit   = True,
    dtype          = None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0.0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)


# ── Environment client ────────────────────────────────────────────────────────
if args.env_url:
    from CarbonSchedulerEnv.client import CarbonSchedulerClient
    env_client = CarbonSchedulerClient(args.env_url)
    print(f"Connected to remote env: {args.env_url} | healthy={env_client.health()}")
else:
    from CarbonSchedulerEnv.server.CarbonSchedulerEnv_environment import CarbonSchedulerEnvEnvironment
    from CarbonSchedulerEnv.models import CarbonSchedulerAction, ScheduleDecision

    class LocalEnvWrapper:
        def __init__(self):
            self.env = CarbonSchedulerEnvEnvironment()
        def reset(self, seed=None):
            obs = self.env.reset(seed=seed)
            return obs.model_dump()
        def step(self, action_json):
            import json
            data = json.loads(action_json) if isinstance(action_json, str) else action_json
            assignments = [ScheduleDecision(**a) for a in data.get("assignments", [])]
            action = CarbonSchedulerAction(assignments=assignments)
            obs = self.env.step(action)
            return obs.model_dump(), obs.reward, obs.done, obs.reward_breakdown

    env_client = LocalEnvWrapper()
    print("Using local in-process environment")


# ── Curriculum config ─────────────────────────────────────────────────────────
CURRICULUM = {
    1: {  # Easy: 3 jobs, 2 regions, no urgent jobs
        "description": "3 jobs, 2 regions (us-west-2 + eu-west-1), no urgent",
        "max_jobs":     3,
        "regions":     ["us-west-2", "eu-west-1"],
        "no_urgent":   True,
        "max_steps":   8,
    },
    2: {  # Medium: 6 jobs, 4 regions, some urgent
        "description": "6 jobs, 4 regions, some urgent",
        "max_jobs":     6,
        "regions":     ["us-west-2", "us-west-1", "eu-west-1", "us-east-1"],
        "no_urgent":   False,
        "max_steps":   16,
    },
    3: {  # Hard: full episode, all 6 regions, mixed priorities
        "description": "Full episode, all 6 regions, mixed priorities",
        "max_jobs":     14,
        "regions":     None,   # all regions
        "no_urgent":   False,
        "max_steps":   24,
    },
}

current_curriculum = CURRICULUM[args.curriculum]
print(f"\nCurriculum level {args.curriculum}: {current_curriculum['description']}\n")


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert carbon-aware AI workload scheduler for a global cloud provider.

Your goal: schedule AI training jobs across data centres to MINIMISE carbon emissions (gCO2) 
while meeting all SLA deadlines.

Key rules:
1. URGENT jobs must be scheduled immediately — never defer them
2. LOW priority jobs should be deferred to low-carbon windows when possible
3. Check carbon forecasts — schedule jobs when renewable energy is high
4. Never overload a data centre beyond its available capacity
5. Oregon (us-west-2) and Ireland (eu-west-1) are typically the cleanest regions

Always respond with ONLY valid JSON in this exact format:
{"assignments": [
  {"job_id": "job_01", "region": "us-west-2", "start_hour": 14},
  {"job_id": "job_02", "defer": true}
]}"""


# ── Rollout function ──────────────────────────────────────────────────────────
def run_episode(seed: int = None) -> Dict[str, Any]:
    """
    Run one full episode and collect (prompt, response, reward) tuples.
    These become the training data for GRPO.
    """
    obs   = env_client.reset(seed=seed)
    done  = False
    total_reward = 0.0
    steps = 0
    trajectory = []

    while not done and steps < current_curriculum["max_steps"]:
        prompt = obs.get("prompt", "")

        # ── Generate action with current model ────────────────────────────
        inputs = tokenizer(
            [f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"],
            return_tensors = "pt",
            truncation     = True,
            max_length     = 1800,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 300,
                temperature    = 0.7,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens = True,
        )

        # ── Step environment ──────────────────────────────────────────────
        obs, reward, done, info = env_client.step(generated)

        trajectory.append({
            "prompt":   prompt,
            "response": generated,
            "reward":   reward,
            "info":     info,
        })

        total_reward += reward
        steps += 1

    return {
        "trajectory":    trajectory,
        "total_reward":  total_reward,
        "steps":         steps,
        "episode_summary": info.get("episode_summary", {}),
    }


# ── Reward function for GRPO ──────────────────────────────────────────────────
def reward_fn(prompts: List[str], completions: List[str], obs_json=None, **kwargs) -> List[float]:
    """
    GRPO reward function — called by GRPOTrainer per batch.
    FIX: uses pre-generated obs rather than resetting env every call (was very expensive).
    Each prompt was generated from env.reset() at dataset build time.
    We replay the completion against a fresh env seeded identically.
    """
    import json as _json
    rewards = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            # Parse the model's JSON output
            start = completion.find("{")
            end   = completion.rfind("}") + 1
            if start < 0 or end <= 0:
                rewards.append(-0.3)   # invalid JSON format
                continue

            data = _json.loads(completion[start:end])

            # Use seed from dataset if available (reproducible eval)
            seed = (i * 7 + 42) % 9999
            env_client.reset(seed=seed)
            _, reward, _, _ = env_client.step(completion[start:end])
            rewards.append(float(reward))

        except _json.JSONDecodeError:
            rewards.append(-0.3)
        except Exception as e:
            print(f"Reward fn error: {e}")
            rewards.append(-0.5)

    return rewards


# ── Build GRPO dataset ────────────────────────────────────────────────────────
def build_prompt_dataset(n: int = 200) -> List[Dict[str, str]]:
    """Generate diverse prompts from environment resets."""
    prompts = []
    for i in range(n):
        seed = i * 7 + args.curriculum * 100
        obs  = env_client.reset(seed=seed)
        prompt_text = obs.get("prompt", "")
        prompts.append({
            "prompt": f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt_text}\n<|assistant|>\n"
        })
    return prompts

print("Building prompt dataset...")
prompt_dataset = build_prompt_dataset(n=150)
print(f"  → {len(prompt_dataset)} prompts generated")


# ── Baseline evaluation ───────────────────────────────────────────────────────
def evaluate_baseline(n_episodes: int = 10) -> Dict[str, float]:
    """Evaluate a simple heuristic baseline for comparison."""
    rewards = []
    for i in range(n_episodes):
        obs  = env_client.reset(seed=i + 5000)
        done = False
        ep_reward = 0.0

        while not done:
            # Naive baseline: always assign to us-west-2 (cleanest region)
            pending = obs.get("jobs_pending", [])
            assignments = []
            for j in pending:
                assignments.append({
                    "job_id":     j["id"],
                    "region":     "us-west-2",
                    "start_hour": obs.get("current_hour", 0),
                })
            naive_action = json.dumps({"assignments": assignments})
            obs, r, done, _ = env_client.step(naive_action)
            ep_reward += r

        rewards.append(ep_reward)

    avg = sum(rewards) / len(rewards)
    print(f"  Naive baseline avg reward over {n_episodes} episodes: {avg:.4f}")
    return {"baseline_reward": avg, "n_episodes": n_episodes}


print("\nRunning baseline evaluation...")
baseline = evaluate_baseline(n_episodes=5)


# ── GRPO Training ─────────────────────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir          = args.output_dir,
    num_train_epochs    = 1,
    per_device_train_batch_size = args.batch_size,
    gradient_accumulation_steps = 2,
    learning_rate       = args.lr,
    max_grad_norm       = 0.3,
    warmup_ratio        = 0.05,
    lr_scheduler_type   = "cosine",
    logging_steps       = 5,
    save_steps          = 50,
    fp16                = not torch.cuda.is_bf16_supported(),
    bf16                = torch.cuda.is_bf16_supported(),
    report_to           = "wandb" if args.log_wandb else "none",
    # GRPO-specific
    num_generations     = 4,       # samples per prompt
    max_completion_length = 300,
    temperature         = 0.7,
)

from datasets import Dataset as HFDataset
hf_dataset = HFDataset.from_list(prompt_dataset)

trainer = GRPOTrainer(
    model          = model,
    args           = training_args,
    train_dataset  = hf_dataset,
    reward_funcs   = [reward_fn],
    processing_class = tokenizer,
)

print(f"\nStarting GRPO training for {args.steps} steps...")
print(f"  Model:      {args.model}")
print(f"  Curriculum: Level {args.curriculum}")
print(f"  Batch size: {args.batch_size}")
print(f"  LR:         {args.lr}")
print()

trainer.train()


# ── Post-training evaluation ──────────────────────────────────────────────────
print("\nPost-training evaluation...")
post_rewards = []
for i in range(10):
    result = run_episode(seed=i + 8000)
    post_rewards.append(result["total_reward"])
    ep_sum = result.get("episode_summary", {})
    print(f"  Episode {i+1}: reward={result['total_reward']:.4f} | "
          f"carbon_saved={ep_sum.get('carbon_saved_gco2', 0):.0f} gCO2 | "
          f"completion={ep_sum.get('completion_rate', 0):.0%}")

post_avg = sum(post_rewards) / len(post_rewards)
improvement = post_avg - baseline["baseline_reward"]
print(f"\nBaseline avg:     {baseline['baseline_reward']:.4f}")
print(f"Post-training avg: {post_avg:.4f}")
print(f"Improvement:      +{improvement:.4f} ({improvement/abs(baseline['baseline_reward'])*100:.1f}%)")


# ── Save model ────────────────────────────────────────────────────────────────
print(f"\nSaving model to {args.output_dir}...")
# IMPORTANT: use merged save path — do NOT upcast 4-bit then merge naively
model.save_pretrained_merged(
    args.output_dir,
    tokenizer,
    save_method = "merged_16bit",
)
print("Done.")

if args.log_wandb:
    wandb.log({
        "baseline_reward":      baseline["baseline_reward"],
        "post_training_reward": post_avg,
        "improvement":          improvement,
    })
    wandb.finish()
