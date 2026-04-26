"""
train.py — TRL + Unsloth GRPO training for CarbonSchedulerEnv

FIXES vs previous version:
  [6] reward_fn now replays against the exact seed used when the prompt was
      generated — so env state matches the prompt the model actually saw.
      Seeds are stored in the dataset alongside prompts, not generated randomly.
      The comment "FIX: uses pre-generated obs" now matches the code.

  [7] LocalEnvWrapper.step() returns a proper info dict containing
      episode_summary so run_episode() can log carbon_saved and completion_rate.
      Previously returned obs.reward_breakdown which has no episode_summary key,
      causing silent empty-dict logging for every episode.

Run in Colab:
  !pip install unsloth trl transformers accelerate datasets
  !python train.py --curriculum 1 --steps 300

Or with remote env:
  python train.py --env_url http://localhost:7860 --steps 500
"""

import argparse
import json
import random
from typing import Any, Dict, List

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
parser.add_argument("--env_url",    default=None)
parser.add_argument("--steps",      type=int, default=300)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr",         type=float, default=5e-6)
parser.add_argument("--output_dir", default="./carbon_scheduler_model")
parser.add_argument("--curriculum", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("--log_wandb",  action="store_true")
args = parser.parse_args()

# ── Model ─────────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import torch

if args.log_wandb:
    import wandb
    wandb.init(project="carbon-scheduler-rl", config=vars(args))

print(f"Loading model: {args.model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = args.model,
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
    print(f"Remote env: {args.env_url} | healthy={env_client.health()}")
else:
    from CarbonSchedulerEnv.server.CarbonSchedulerEnv_environment import CarbonSchedulerEnvEnvironment
    from CarbonSchedulerEnv.models import CarbonSchedulerAction, ScheduleDecision

    class LocalEnvWrapper:
        """
        FIX [7]: step() now returns a proper 4-tuple where info is a real dict
        containing episode_summary when done=True, not reward_breakdown.
        Previously returned obs.reward_breakdown as info, causing
        info.get("episode_summary", {}) to silently return {} every time.
        """
        def __init__(self, curriculum_stage: int = 1):
            self.env = CarbonSchedulerEnvEnvironment(curriculum_stage=curriculum_stage)

        def reset(self, seed=None) -> dict:
            obs = self.env.reset(seed=seed)
            return obs.model_dump()

        def step(self, action_json: str):
            data = json.loads(action_json) if isinstance(action_json, str) else action_json
            assignments = [ScheduleDecision(**a) for a in data.get("assignments", [])]
            action = CarbonSchedulerAction(assignments=assignments)
            obs = self.env.step(action)

            # FIX [7]: build proper info dict so callers can read episode_summary
            info = {
                "reward_breakdown": obs.reward_breakdown,
                "episode_summary":  obs.episode_summary,   # populated when done=True
                "curriculum_stage": obs.curriculum_stage,
            }
            return obs.model_dump(), obs.reward, obs.done, info

    env_client = LocalEnvWrapper(curriculum_stage=args.curriculum)
    print(f"Local env | curriculum stage={args.curriculum}")


# ── Curriculum ────────────────────────────────────────────────────────────────
CURRICULUM = {
    1: {"description": "3 jobs, 2 regions, no urgent",            "max_steps": 8},
    2: {"description": "6 jobs, 4 regions, some urgent",          "max_steps": 16},
    3: {"description": "14 jobs, all 6 regions, mixed priorities", "max_steps": 24},
}
current_curriculum = CURRICULUM[args.curriculum]
print(f"Curriculum {args.curriculum}: {current_curriculum['description']}\n")


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert carbon-aware AI workload scheduler for a global cloud provider.

Your goal: schedule AI training jobs across data centres to MINIMISE carbon emissions (gCO2)
while meeting all SLA deadlines.

Key rules:
1. URGENT jobs must be scheduled immediately — never defer them
2. LOW priority jobs: check the carbon forecast. If a cleaner window is coming in the next
   few hours, defer them. If carbon is already low, assign now.
3. Read the forecast — California is cleanest midday (solar). Oregon is cleanest overnight.
4. Never overload a data centre beyond its available GPU-hr capacity.

Always respond with ONLY valid JSON:
{"assignments": [
  {"job_id": "job_01", "region": "us-west-2", "start_hour": 14},
  {"job_id": "job_02", "defer": true}
]}"""


# ── Rollout ───────────────────────────────────────────────────────────────────
def run_episode(seed: int = None) -> Dict[str, Any]:
    obs   = env_client.reset(seed=seed)
    done  = False
    total = 0.0
    steps = 0
    info  = {}

    while not done and steps < current_curriculum["max_steps"]:
        prompt = obs.get("prompt", "")
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
            skip_special_tokens=True,
        )

        obs, reward, done, info = env_client.step(generated)
        total += reward
        steps += 1

    # FIX [7]: episode_summary is now correctly populated in info
    return {
        "total_reward":    total,
        "steps":           steps,
        "episode_summary": info.get("episode_summary", {}),
    }


# ── Dataset — seeds stored alongside prompts (fix [6]) ───────────────────────
def build_prompt_dataset(n: int = 150) -> List[Dict[str, Any]]:
    """
    FIX [6]: Store the seed alongside each prompt so reward_fn can replay the
    exact same env state when evaluating completions. Without this, reward_fn
    was resetting to a different random seed than the one used to generate the
    prompt, making the env state inconsistent with what the model saw.
    """
    prompts = []
    for i in range(n):
        seed = i * 7 + args.curriculum * 100
        obs  = env_client.reset(seed=seed)
        prompts.append({
            "prompt": (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{obs.get('prompt', '')}\n"
                f"<|assistant|>\n"
            ),
            "seed":   seed,   # FIX [6]: stored for faithful reward replay
        })
    return prompts

print("Building prompt dataset...")
prompt_dataset = build_prompt_dataset(n=150)
print(f"  → {len(prompt_dataset)} prompts")


# ── Reward function (fix [6]) ─────────────────────────────────────────────────
def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    FIX [6]: Reset env with the seed that was used when the prompt was generated.
    Previously used a random seed (i * 7 + 42) unrelated to dataset generation,
    so the environment state did not match what the model saw in its prompt.
    Now: extract seed from dataset via kwargs["seed"] if available, else fall back.
    """
    # GRPOTrainer passes extra dataset columns through kwargs
    seeds = kwargs.get("seed", [None] * len(prompts))

    rewards = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            # Extract JSON from completion (model may wrap in prose)
            start = completion.find("{")
            end   = completion.rfind("}") + 1
            if start < 0 or end <= 0:
                rewards.append(-0.3)
                continue

            action_str = completion[start:end]
            json.loads(action_str)  # validate JSON before sending

            # FIX [6]: replay with the exact seed from dataset
            seed = seeds[i] if seeds[i] is not None else (i * 13 + 99)
            env_client.reset(seed=int(seed))
            _, reward, _, _ = env_client.step(action_str)
            rewards.append(float(reward))

        except json.JSONDecodeError:
            rewards.append(-0.3)
        except Exception as e:
            print(f"reward_fn error at i={i}: {e}")
            rewards.append(-0.5)

    return rewards


# ── Baseline ──────────────────────────────────────────────────────────────────
def evaluate_baseline(n: int = 5) -> Dict[str, float]:
    """
    Naive baseline: always assign every job to us-west-2 immediately.
    With the fixed naive_carbon baseline (best-region-now), this will score
    ~0 carbon_score (since us-west-2 IS already the best region most of the time).
    This makes the baseline much harder to beat and the improvement curves real.
    """
    rewards = []
    for i in range(n):
        obs  = env_client.reset(seed=i + 5000)
        done = False
        ep_r = 0.0
        while not done:
            pending = obs.get("jobs_pending", [])
            asgn    = [{"job_id": j["id"], "region": "us-west-2",
                        "start_hour": obs.get("current_hour", 0)} for j in pending]
            obs, r, done, _ = env_client.step(json.dumps({"assignments": asgn}))
            ep_r += r
        rewards.append(ep_r)
    avg = sum(rewards) / len(rewards)
    print(f"  Naive baseline avg reward ({n} eps): {avg:.4f}")
    return {"baseline_reward": avg}

print("\nBaseline evaluation...")
baseline = evaluate_baseline()


# ── GRPO training ─────────────────────────────────────────────────────────────
from datasets import Dataset as HFDataset

hf_dataset = HFDataset.from_list(prompt_dataset)

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
    num_generations     = 4,
    max_completion_length = 300,
    temperature         = 0.7,
)

trainer = GRPOTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = hf_dataset,
    reward_funcs     = [reward_fn],
    processing_class = tokenizer,
)

print(f"\nStarting GRPO | curriculum={args.curriculum} | steps={args.steps}\n")
trainer.train()


# ── Post-training eval ────────────────────────────────────────────────────────
print("\nPost-training evaluation...")
post_rewards = []
for i in range(10):
    result = run_episode(seed=i + 8000)
    post_rewards.append(result["total_reward"])
    s = result["episode_summary"]   # FIX [7]: now correctly populated
    print(f"  Ep {i+1}: reward={result['total_reward']:.4f} | "
          f"carbon_saved={s.get('carbon_saved_gco2', 0):.0f} gCO2 | "
          f"completion={s.get('completion_rate', 0):.1%} | "
          f"carbon_eff={s.get('carbon_efficiency', 0):.1%}")

post_avg    = sum(post_rewards) / len(post_rewards)
improvement = post_avg - baseline["baseline_reward"]
print(f"\nBaseline:      {baseline['baseline_reward']:.4f}")
print(f"Post-training: {post_avg:.4f}")
print(f"Improvement:   +{improvement:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {args.output_dir}...")
# Use merged save — do NOT upcast 4-bit model then naive-merge LoRA
model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")
print("Done.")

if args.log_wandb:
    wandb.log({"baseline_reward": baseline["baseline_reward"],
               "post_training_reward": post_avg, "improvement": improvement})
    wandb.finish()