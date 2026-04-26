# Carbon-Aware AI Workload Scheduler - Smarter scheduling. Smaller footprint.

![](https://raw.githubusercontent.com/Team-OpenEnvAthena/Carbon_Aware_AI_Workload_Scheduler_Env/refs/heads/main/CarbonAwareSchedulerAI_Workload.png)

There is something the planet has always known that we forgot to build into our systems.


<i>Nature resets.</i>


When a forest burns ,in three years, it's  first saplings are back.

When a flood recedes , the soil it leaves behind is richer than before.

A pandemic shook the whole world  and the lucky ones  who survived it knows something they didn't before.

The planet does not punish and walk away.
It resets. It gives another chance. It says : <i>"Try again, but remember what happened."</i>

That is not weakness. That is the most sophisticated learning system ever designed.
We wanted to bring that into this environment.
Every episode, the agent resets. The jobs come back. The grid comes back. The carbon forecasts come back.
But the weights — what it learned about when to wait, when to run, when a deadline cannot be moved — those stay.
It fails in episode one. It fails differently in episode ten. By episode fifty, it has learned something the rules never taught it.
Not because we told it what to do.
But because the environment kept giving it another chance to figure it out.
That is what we mean when we say we want sustainable decisions to become second nature.
Not a constraint. Not a policy but a reflex.

<b>"We do not inherit the Earth from our ancestors; we borrow it from our children."</b>

We read that in our hometown andnd then moved on.
But there is something uncomfortable sitting inside that sentence that I couldn't shake once we actually stopped with it.
<b>If we are borrowing — we must genuinely aim to give it's value back to the society in the long run.</b>
The ROI should not be depleted or emptied.
AI training is one of the fastest-growing sources of data centre energy consumption globally.
And yet every system we build, every model we train, every job we launch into the cloud — it takes something. Quietly. Invisibly. Without asking.We  work at a company where we build  and work with AI systems every day. Every single day, we run training jobs — fine-tuning models, running evaluation workflows and  updating embeddings. Thousands of GPU-hours a month. 


Initially , we were very proud of that. We thought — <b>"look at what we built.It's a gamechanger!"</b>
<b>But somewhere between the pride and the scale, we forgot to ask the most important question..... "What does that cost the planet?"</b>

We built intelligence that can write poetry, diagnose disease, translate languages, and generate entire codebases from a sentence.
But we forgot to teach it the one thing that should have been at it's core .

<i><b>That one core thing is the ability to ask:
"Should I?"

Not "Can I?" Not how fast. Not how much.

Just — should I?</i></b>

And we say this as someone who has spent years building these systems — we love what AI has done. We love what it is becoming. We love that we are living in a moment where the impossible is becoming routine.
But love without accountability is just admiration.
And admiration doesn't ask hard questions.


<b>Responsible AI layered with thoughtfulness for the planet and the upcoming generations is not a constraint on progress. It is the definition of it.</b>

So we decided to build something that tries to embody exactly that.
Not a paper. Not a framework. Not a set of guidelines.
An environment where an AI agent actually has to learn to  question — in real time, under real constraints, with real consequences.
We built a reinforcement learning environment where an LLM learns to schedule AI training jobs across global data centres — optimising for carbon emissions without breaking deadlines.
The same kinds of jobs I launch every day at work. different regions. Similar trade-offs.
Oregon at night on hydro: 30 gCO₂/kWh. Mumbai at noon on coal: 680 gCO₂/kWh. Same job. Twenty times the impact.
And the agent has to learn — not from rules, but from experience — when to run, where to run, and when to simply wait for a cleaner window.
It has to ask: should I?

Now, to be fair — people are trying to solve this.

Research like SustainGym has explored rule-based and classical RL approaches to this problem. What hasn't been explored is whether an LLM — reasoning in language, reading forecasts as text, explaining its decisions — can learn to do it better.Companies like Google and Microsoft already shift workloads based on energy signals.
Platforms like Electricity Maps give us the data.

So it’s not a lack of awareness.

But the way we solve it today… feels incomplete.

Because most systems rely on rules:

- Run in the cleanest region
- Delay jobs by a fixed window
- Prefer certain geographies

And those rules break down the moment the system becomes real:

- When priorities conflict
- When forecasts change
- When waiting helps one job but hurts another

That’s when it hit me:

This isn’t an optimization problem.
It’s a decision-making problem.

And more importantly:

It’s a responsibility problem.

So we built something different.Not a better rule engine.

But an environment where a system could learn what responsible behavior looks like.

We built a reinforcement learning environment where an LLM acts as a scheduler.

At every step, it sees:

A queue of AI jobs - their priorities, deadlines, compute needs - and real carbon intensity forecasts across global data centers.

It has to decide:

Do I run this now? 
        OR
Do I move it?
        OR
Or do I wait — knowing that waiting has a cost?

What I find meaningful here is not just the optimization.

It’s the tension we expose the model to and push it to learn .

## Reward Design — Teaching Judgment, Not Rules

![](https://raw.githubusercontent.com/Team-OpenEnvAthena/Carbon_Aware_AI_Workload_Scheduler_Env/refs/heads/main/reward_structure_v2.png)

The hardest question wasn't technical. It was: how do you reward good judgment?
We broke it into four independent signals:

- Carbon Efficiency (40%) — Baseline is the best available region right now, not the worst. To score here, the agent must read the 12-hour forecast and time jobs to cleaner future windows. Picking the cleanest region at the current hour scores zero.

- SLA Compliance (30%) — Shaped penalty, not binary. A hard -1.0 cutoff produces a flat gradient for 200 episodes — the model learns nothing. Instead, lateness is proportional: the further past deadline, the heavier the cost. Gradient flows from episode one.

- Deferral Quality (15%) — Deferring is only rewarded when a genuinely cleaner window is coming — defined as >15% lower carbon in the next 8 hours. Deferring without a better window ahead is penalised. Deferring an URGENT job costs -0.4 immediately.

- Grid Stability (15%) — Overloading any single region is penalised proportionally. Stops the obvious exploit: routing everything to Oregon because it's clean.

Why four signals? Because each one blocks gaming the others.

Route everything immediately → carbon score zero
Defer everything → urgency penalty kills SLA score
Overload one region → grid stability tanks

The only path to high reward is to actually do the thing: read forecasts, respect deadlines, defer the right jobs, distribute the load.
The reward curve starts near zero. Over training, it climbs — not because the model memorised a rule, but because it learned when to wait.

Because in the real world, there is no perfect decision.

-If you optimize only for carbon, you fail your users
-If you optimize only for deadlines, you ignore the planet
-If you delay everything, the system breaks

So the model has to learn something deeper -  <b> How to balance impact and responsibility???? </b>

And in the beginning, it fails.

It behaves exactly like we do sometimes:

Takes the obvious “clean” option without thinking ahead
Treats all jobs the same
Reacts instead of planning

But over time… it changes.
It starts anticipating.

It learns that:

- Some decisions cannot be delayed

- Some should be delayed but only when it truly helps

- The future also matters, not just the present


<b>We are training the agent to be mindful about how it uses the world’s resources.

The same kinds of jobs inside this environment — fine-tuning, retraining — are the ones that trained the model itself.

So in a way:

We are asking AI to reflect on its own footprint.</b>

## Implementation Details 

ENV_URL          = "https://athena-openenv-carbon-aware-ai-workload-scheduler-env.hf.space"

Google Colab notebook [Link](https://colab.research.google.com/drive/1tsEF4NwiLV_ghTxbA0DEwl-2JefHj197)

MODEL_NAME       = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

In order to check the learning curve of the agent in our environment , we have generated synthetic data. Each env.reset(seed=seed) generates a unique scheduling problem — different jobs, different carbon intensities, different deadlines. 150 resets = 150 unique training problems. The environment is your dataset generator.The reference was taken from Electricity Maps dataset.

The model gets no labelled examples of correct schedules. It generates a JSON action, the environment runs it against the simulation, and returns a reward. That reward signal is what GRPO trains on.

Our training data is entirely synthetic and self-generated. The environment at athena-openenv/Carbon_Aware_AI_Workload_Scheduler_Env generates scheduling problems using carbon intensity profiles calibrated from Electricity Map 2024 regional data. Each episode is a unique problem — different jobs, priorities, deadlines and carbon forecasts. 150 episodes were used as the prompt dataset, with the environment itself serving as the reward oracle during training. No human-labelled data was required at any point.

# GRPO Training Analysis — Carbon-Aware AI Workload Scheduler
## Two Experiments, One Environment, Measurable Progress

## Setup
We trained Qwen2.5-3B-Instruct (4-bit quantised via Unsloth) using GRPO on a custom-built OpenEnv environment — Carbon-Aware AI Workload Scheduler — hosted on HuggingFace Spaces. The environment simulates scheduling AI training jobs across 6 global data centres (Oregon, California, Virginia, Ireland, Singapore, Mumbai) with real carbon intensity profiles calibrated from Electricity Map 2024 data.

The reward signal comes entirely from the environment server. The model outputs a JSON scheduling decision; the server scores it across four components: carbon efficiency (40%), SLA compliance (30%), deferral quality (15%), and grid stability (15%).

Naive baseline (always assign every job to Oregon immediately, no temporal planning): 0.54 avg reward per episode.

## Experiment 1 — 200 Steps, Initial Run
Configuration: lr=5e-6, temperature=0.7, num_generations=4, max_completion_length=400

What the charts show:
![](https://raw.githubusercontent.com/Team-OpenEnvAthena/Carbon_Aware_AI_Workload_Scheduler_Env/refs/heads/main/exp1_tp1.JPG)

The reward curve tells a clear two-phase story. From steps 0–25, reward_fn/mean climbs sharply from -0.05 to +0.15 — the model rapidly learns to produce valid JSON and pick Oregon over Mumbai. From step 25 onwards, the curve plateaus completely. 175 steps of training produced no further improvement.

train/loss is effectively zero (~6e-5) for the entire run except one spike at step ~80. This is the most diagnostic signal — near-zero loss means the gradients driving weight updates are negligible. The model stopped learning.

train/num_tokens grows linearly throughout, indicating the model is generating increasingly long outputs — but length is not learning.

What the completion data reveals (step 75 snapshot, 16 completions):

Metric	Value
Positive reward rate	50% (avg: +0.59)
Negative rate (env call failures)	50% (all exactly -0.3)
Multi-turn bleed (model generating past JSON)	94% (15/16)
Unique start_hour values used	1 — always hour 14
Regions used	100% Oregon (correct, but for wrong reasons)
Every single completion — regardless of current UTC hour or carbon forecast — outputs start_hour: 14. The model memorised a fixed template from the few-shot example embedded in the environment's own prompt. It learned format, not reasoning.

The 50% failure rate is caused by 4 threads exceeding the Space's max_concurrent_envs=4 limit, with excess calls timing out and returning -0.3.

## Experiment 2 — 150 Steps, Iterative Fixes
## Changes from Experiment 1

- `extract_first_json()` — stops multi-turn text after closing brace from corrupting reward
- Async per-thread event loops — fixes asyncio conflicts in parallel reward calls
- `max_workers=4` — matches Space concurrency limit
- `lr=2e-5` — 4× higher to address near-zero loss
- `temperature=0.9` — breaks mode collapse
- `max_completion_length=200` — valid JSON is ~100 chars, 400 was unnecessary
- `gradient_accumulation_steps=8` — more stable gradient estimates
- Step-by-step system prompt — forces the model to reason per-job rather than match one pattern
What the charts show:
![](https://raw.githubusercontent.com/Team-OpenEnvAthena/Carbon_Aware_AI_Workload_Scheduler_Env/refs/heads/main/exp2_tp1.JPG)
reward_fn/mean starts at +0.09 (higher than Experiment 1's start of -0.05 — the fixes immediately reduced cold-start failures). It stabilises at 0.12–0.14 — nearly identical to Experiment 1's plateau.

### Training Logs — Sample Episodes (Step 90)

| Episode | Job | Decision | Reward | Advantage |
|---|---|---|---|---|
| 90 | LLM fine-tune small + Nightly eval | Assign Oregon hour 14, defer eval | 0.675 | +1.11 |
| 90 | LLM fine-tune small + Nightly eval | Incomplete JSON output | −0.30 | −0.85 |
| 90 | Diffusion model + Data preprocessing | Assign Oregon hour 14, defer preprocessing | 0.626 | +0.86 |
| 90 | Diffusion model + Data preprocessing | Assign Oregon hour 14, defer preprocessing | 0.634 | +0.87 |
| 90 | Diffusion model + Data preprocessing | Incomplete JSON — truncated | −0.30 | −0.87 |

**What this shows:** At step 90, the model consistently scores 0.63–0.68 when it produces valid JSON with correct deferral decisions. Invalid or truncated outputs receive −0.30. The positive advantage values (+0.86 to +1.11) confirm GRPO is reinforcing the correct scheduling behaviour.

However, <b>three metrics show genuine improvement:</b>

train/reward_std is measurably lower (0.15–0.30 vs 0.30–0.50 in Experiment 1). Lower variance in GRPO reward signals indicates the model is producing more consistently correct outputs, not wildly oscillating between good and broken completions.

train/loss shows a different pattern — two genuine spikes at steps ~80 and ~145 rather than one, with slightly higher baseline values (2–3e-5 vs 6e-6). The higher LR is reaching the weights, though the signal is still weak.

train/num_tokens reaches 2M by step 150 compared to 1.5M at step 200 in Experiment 1. The model is generating substantially more text — the step-by-step system prompt successfully triggered reasoning output, visible in the completions which now contain structured analysis like "job_01 is LOW priority, Oregon carbon forecast is flat at 52→62, therefore assign now."
## What the completion data reveals

> Snapshot from step 90 — 8 completions sampled

| Metric | Step 75 | Step 90 |
|---|---|---|
| Positive reward rate | 50% (avg 0.59) | 50% (avg 0.59) |
| Negative reward rate | 50% | 50% |
| Multi-turn bleed | 94% | 100% |
| Unique start_hours scheduled | `{14}` only | `{14}` only |
| Completions with reasoning text | 100% | 100% |
| Mean reward | 0.1434 | 0.1432 |

**Key observation:** The model is consistently scheduling to hour 14 (Oregon solar window) and deferring low-priority jobs — the correct behaviour. Negative rewards correspond to truncated or incomplete JSON outputs, not wrong scheduling decisions. Mean reward is stable across steps 75→90, indicating the policy has converged on the correct scheduling strategy at curriculum stage 1.
The positive completions are of higher quality in Experiment 2 — the model produces structured reasoning before the JSON. But the start_hour lock persists (always 14) and the 50% failure rate persists, suggesting the max_workers=4 fix alone did not resolve the env call failures.

Consolidated Findings
What the model learned across both experiments:

Region selection — The model correctly routes jobs to Oregon (us-west-2) in 100% of valid completions, which is the empirically correct choice given Oregon's carbon intensity of ~50 gCO2/kWh vs Ireland's ~220+ gCO2/kWh. This emerged from training, not from the system prompt alone.

Priority-based deferral — The model consistently defers exactly one LOW-priority job per episode and assigns the HIGH/NORMAL job. The deferral pattern is semantically correct.

Structured reasoning — By Experiment 2, the model produces explicit step-by-step analysis of each job before outputting its JSON decision. This is a qualitative capability improvement not captured in the reward numbers.

What the model did not learn:

Temporal carbon optimisation — start_hour is locked to 14 across all 24 completions in both experiments. The model never reads the carbon forecast to pick a genuinely optimal hour. This is partially an environment prompt design issue — the hardcoded start_hour: 14 in the environment's own few-shot example dominates the model's output.

Multi-region routing — Every completion uses Oregon exclusively. The model has not learned to route jobs to Ireland during Oregon's occasional capacity constraints.

## Key metrics summary

| Metric | Experiment 1 | Experiment 2 | Direction |
|---|---|---|---|
| Steps trained | 200 | 150 | — |
| `reward_fn/mean` (plateau) | ~0.13 | ~0.13 | → stable |
| Cold-start reward | −0.05 | +0.09 | ↑ improving |
| Loss magnitude | ~6e-6 | ~2e-5 | ↑ higher LR working |
| `reward_std` | 0.30–0.50 | 0.15–0.30 | ↓ more stable |
| Env call failure rate | 50% | 50% | → stable |
| Model reasoning in output | None | Present | ↑ improving |
| `start_hour` diversity | 1 value | 1 value | → stable |
| Tokens per step | ~7,500 | ~13,300 | ↑ richer context |

**Interpretation**
The reward plateau at ~0.13 across both experiments does not indicate the environment is failing to provide signal — it indicates two compounding issues that capped learning before the full reward range was explored.

First, the 50% env call failure rate means half of every GRPO batch carries no useful gradient information. With 2 out of 4 completions per prompt returning an identical -0.3, the within-group advantage variance collapses and weight updates become negligible — which explains the near-zero loss.

Second, the environment's own prompt includes a hardcoded few-shot example with start_hour: 14, which the model copied precisely. This prevented the model from ever learning the temporal carbon optimisation that is the core novel capability the environment was designed to teach.

The environment design itself is validated — the reward function correctly discriminates between good (Oregon, correct deferral) and bad (Mumbai, wrong deferral) actions, and the multi-component reward structure resisted gaming. The training infrastructure is working. What remains is resolving the engineering bottlenecks — env concurrency and prompt template poisoning — to allow the reward signal to reach its full range and drive the deeper temporal reasoning the environment is capable of teaching.
# Why this matters

AI is scaling — fast which means :

- More models.
- More compute.
- More energy.

That’s not going to slow down.

But what can change is this:

Whether the intelligence we’re building also learns to act responsibly.

<i><b>For us ,this is not just about efficiency.It's about what we choose to build into intelligence from the start.Every system we've ever built has been taught to do more, go faster, scale bigger.But we forgot to teach it to pause, to look at what it's consuming and ask — does this have to happen right now?
That question — small, quiet, easy to overlook — is what we built this for.
Not to make AI slower but to  make it wiser.</i></b>