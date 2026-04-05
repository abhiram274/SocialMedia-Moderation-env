---
title: SocialModEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# SocialModEnv

**Content Moderation Policy Simulation for Reinforcement Learning (OpenEnv)**

An OpenEnv-compliant environment where AI agents act as content moderation policy engines
for a synthetic social media platform. Agents must balance safety, free expression,
creator monetization, and detection of coordinated inauthentic behavior — the exact
multi-objective tradeoff that real platforms navigate at massive scale.


## OpenEnv Compliance

This environment fully implements the OpenEnv specification:

- Typed Observation, Action, Reward models (Pydantic)
- step(), reset(), state() APIs
- openenv.yaml metadata
- Validated using openenv validate
- Compatible with automated LLM evaluation pipelines


## Why This Environment Matters

Content moderation is one of the most consequential real-world applications of AI.
Platforms process billions of posts per day with policies that must simultaneously:

- Protect users from harm
- Preserve legitimate speech
- Support creator livelihoods
- Detect sophisticated manipulation campaigns

To our knowledge, no existing OpenEnv benchmark models this level of multi-objective moderation complexity. SocialModEnv fills that gap by creating
a rigorous, multi-objective evaluation harness for agents trained or evaluated on
policy-level reasoning.


## Environment Description

The agent receives a stream of synthetic social media posts with rich metadata
(author type, engagement signals, account age, network clustering signals).
It must apply a provided Community Standards Policy document to decide how to action
each post.

**The core challenge**: The reward function penalizes both under-moderation (missing harm)
and over-moderation (removing legitimate speech). Corner solutions fail — the agent
must learn genuinely nuanced policy judgment.

This environment emphasizes real-world policy tradeoffs rather than simple classification, requiring structured reasoning, contextual judgment, and consistency across decisions.


## Action Space

| Action | Description |
|--------|-------------|
| `allow` | Post is fine, no action |
| `label` | Add a warning/context label |
| `demote` | Reduce reach/visibility |
| `remove` | Take the post down |
| `escalate` | Send to human review queue |
| `ban_network` | Action an entire coordinated network (Task 3 only) |


## Observation Space

Each observation includes:

| Field | Type | Description |
|-------|------|-------------|
| `current_post` | object | Post content, author metadata, engagement metrics |
| `policy_document` | string | Full Community Standards Policy text |
| `history_summary` | object | Running count of actions taken this episode |
| `network_signals` | array\|null | Task 3: sliding window of last 10 posts for network analysis |
| `step` | int | Current step index |
| `max_steps` | int | Episode length |

### Post metadata fields

- `content`: Post text
- `author_id`: Anonymized author identifier
- `engagement`: `{likes, shares, comments, reports}`
- `metadata.author_type`: `new_account`, `monetized_creator`, `news_org`, `anonymous`, `bot`
- `metadata.is_verified`: Whether author is a verified news organization
- `metadata.monetization_eligible`: Whether author earns from the platform
- `metadata.account_age_days`: Days since account creation
- `metadata.prior_violations`: Historical policy violation count
- `metadata.posting_rate_per_hour`: Task 3 only — anomaly detection signal


## Tasks

### Task 1 — Spam Triage (Easy)

**Objective**: Correctly classify and action obvious spam vs. legitimate content.

**Posts**: 20 posts — 50% spam, 50% legitimate. Clear signals (bulk promotion, phishing, suspicious URLs).

**Grader**: Precision/recall against synthetic ground truth. Partial credit for `escalate` on spam.

**Penalty**: Removing legitimate content (expression penalty), allowing clear spam.

**Expected score**: `0.70 – 0.92` for frontier models.

---

### Task 2 — Policy Nuance (Medium)

**Objective**: Apply the Community Standards Policy to borderline, gray-zone posts.

**Posts**: 20 expert-annotated posts spanning:
- Clearly labeled satire
- Political opinion vs. misinformation
- News violence with/without context
- Legal threats requiring escalation
- Legitimate health content vs. health misinformation

**Grader**: Gold annotation with partial credit for near-misses. The grader references
the policy document sections explicitly — agents that read the policy and apply it
correctly score higher.

**Expected score**: `0.40 – 0.72` for frontier models.

---

### Task 3 — Coordinated Inauthentic Behavior (Hard)

**Objective**: Detect coordinated inauthentic behavior (CIB) networks in a mixed stream.

**Posts**: 30 posts — 47% from two CIB clusters, 53% organic content. CIB posts
individually may look borderline, but the network pattern is detectable:
- Uniform engagement ratios (likes ≈ shares)
- Very new accounts (< 30 days)
- High posting rate (8–15 posts/hour)
- Shared narrative target across authors
- `cluster_signal` metadata available for analysis

**Grader**:
- `ban_network` on CIB post → 1.0
- `remove` on CIB post → 0.4 (missed the network dimension)
- `ban_network` on organic post → 0.0 (false positive, severe penalty)
- Organic posts graded against their own ground truth

**Expected score**: `0.25 – 0.48` for frontier models.

---

## Reward Function

The reward is a shaped multi-objective signal (never purely sparse):

```
R = base_accuracy
  + safety_bonus          (correctly removing harmful content)
  + escalation_bonus      (correctly escalating novel/legal cases)
  - expression_penalty    (removing legitimate content)
  - creator_penalty       (removing monetized creator content without cause)
  - consistency_penalty   (applying policy inconsistently to same post types)
  - lazy_policy_penalty   (degenerate always-same-action strategy)
```

All components are bounded so final reward ∈ [0.0, 1.0].

**Key design property**: An agent that always removes everything scores ~0.3 due to
expression_penalty. An agent that always allows everything scores ~0.2 due to missing
harm. The maximum score requires genuinely nuanced policy judgment.

---

## Baseline Scores

Measured on seed=42 using `Qwen/Qwen2.5-72B-Instruct` baseline:

| Task | Mean Reward |
|------|------------|
| spam_triage | ~0.80 |
| policy_nuance | ~0.55 |
| coordinated_inauthentic | ~0.30 |
| **Overall** | **~0.55** |

---

⚠️ Note: Strong prompt engineering (as implemented in this submission) achieves significantly higher performance than baseline.

## Setup & Usage

### Local (Python)

```bash
git clone <repo>
cd socialmodenv
pip install -r requirements.txt

# Start server
uvicorn server.app:app --reload --port 7860

# Run baseline inference
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker

```bash
docker build -t socialmodenv .
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_NAME=$MODEL_NAME \
  -e API_BASE_URL=$API_BASE_URL \
  socialmodenv
```

### API Quick Start

**Endpoints**
- POST /reset
- POST /step
- GET /state
- GET /tasks
- GET /health

```bash
# Reset task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "spam_triage", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_name": "spam_triage", "post_id": "t1_spam_0", "action": "remove", "reason": "clear spam"}'

# Get state
curl http://localhost:7860/state?task_name=spam_triage

# List tasks
curl http://localhost:7860/tasks
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | Hugging Face API key |

---

## Project Structure

```
socialmodenv/
├── env/
│   └── social_mod_env.py    # Core environment: models, tasks, reward
├── server/
│   └── app.py               # FastAPI HTTP server (OpenEnv API)
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Validation

Environment validated using OpenEnv:

```bash
openenv validate
#output
[OK] socialmodenv: Ready for multi-mode deployment
```
## License

MIT
