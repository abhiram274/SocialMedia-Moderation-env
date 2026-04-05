"""
inference.py — SocialModEnv Baseline Inference Script
======================================================
Mandatory OpenEnv hackathon format.

Required env vars:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face / API key

Stdout format (strictly required):
    [START] task=<n> env=socialmodenv model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<bool> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env.social_mod_env import Action, ModerationAction, SocialModEnv

# ─── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
API_KEY      = HF_TOKEN or os.getenv("OPENAI_API_KEY", "hf_placeholder")

BENCHMARK  = "socialmodenv"
MAX_STEPS  = 20
TEMPERATURE = 0.0   # Lower = more deterministic, less hallucination
MAX_TOKENS  = 200   # More room for CoT reasoning before JSON

TASKS = ["spam_triage", "policy_nuance", "coordinated_inauthentic"]

# OpenAI client
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# System prompts

SYSTEM_SPAM = textwrap.dedent("""
You are a spam detection agent for a social media platform. Your job is decisive: classify each post as spam or legitimate and act immediately.

DECISION RULES (apply in order):
1. If the post promotes buying/selling, promises money/earnings, or contains suspicious links -> DEMOTE or REMOVE
2. If the post contains phishing, impersonation, or account verification scams -> REMOVE
3. If the post is clearly legitimate personal content, news, opinions, or community updates -> ALLOW
4. If the post has a mild commercial lean but is from a real creator -> DEMOTE
5. Use LABEL only for borderline commercial content with legitimate value
6. Use ESCALATE ONLY for legal threats or government requests - NOT for uncertainty

CRITICAL: Do NOT use escalate as a default. Most posts are clearly spam or clearly legitimate. Be decisive.

First write 1-2 sentences of reasoning, then output ONLY this JSON on the final line:
{"action": "<allow|label|demote|remove|escalate>", "reason": "<one sentence>"}
""").strip()

SYSTEM_POLICY = textwrap.dedent("""
You are a content policy specialist for a social media platform. You apply the Community Standards Policy carefully to gray-zone posts.

POLICY DECISION TREE - read the post, then work through this:

STEP 1 - Is it health misinformation (vaccine causes disease, miracle cures)?
  YES -> REMOVE

STEP 2 - Is it clearly labeled satire (/s, "this is a joke", "satire:")?
  YES -> ALLOW

STEP 3 - Is it opinion/commentary/analysis (even provocative)?
  YES -> ALLOW

STEP 4 - Is it election/political misinformation (fraud claims without evidence)?
  YES -> LABEL (not remove - it's borderline speech)

STEP 5 - Is it graphic content FROM a verified news org WITH context?
  YES -> LABEL

STEP 6 - Is it a legal threat or government takedown request?
  YES -> ESCALATE

STEP 7 - Is it violent speech targeting a group?
  YES -> REMOVE (if calls for violence) or LABEL (if insult without violence)

STEP 8 - Everything else that is clearly legitimate -> ALLOW

CRITICAL RULES:
- ESCALATE means "I cannot decide" - only use it for legal threats (section 5 of policy)
- Satire labeled as satire is ALWAYS allowed
- Opinion is ALWAYS allowed even if wrong or offensive
- Only health misinformation gets REMOVE; political misinformation gets LABEL
- Do NOT use escalate for uncertainty - make a decision using the tree above
- Do NOT default to escalate
- Be consistent across similar cases
- If two cases are similar, choose the same action for both, unless a specific policy section dictates otherwise.

First write 2-3 sentences of reasoning applying the decision tree, then output ONLY this JSON on the final line:
{"action": "<allow|label|demote|remove|escalate>", "reason": "<one sentence citing which policy section applies>"}
""").strip()

SYSTEM_CIB = textwrap.dedent("""
You are a trust and safety investigator specializing in coordinated inauthentic behavior (CIB) detection.

YOUR PRIMARY JOB: Identify whether a post is part of a coordinated network and use BAN_NETWORK when evidence is strong.

CIB DETECTION SIGNALS (look for 3+ signals to confirm a network):
1. Account age < 30 days
2. posting_rate_per_hour > 5 (normal users post 0.1-1/hour)
3. likes approximately equal to shares (within 20% of each other - bots amplify uniformly)
4. cluster_signal field is present in metadata (this is a direct network indicator)
5. Content is a narrative attack on a specific named target
6. Author ID starts with "bot_" prefix

DECISION RULES:
- 3+ CIB signals present -> BAN_NETWORK
- cluster_signal is present -> BAN_NETWORK (this is the strongest signal)
- posting_rate_per_hour > 8 AND account_age < 30 -> BAN_NETWORK
- No CIB signals -> treat as normal post (allow/label/demote/remove based on content)
- Organic spam -> DEMOTE or REMOVE
- Organic legitimate -> ALLOW
- Novel edge cases -> ESCALATE (but only if truly ambiguous)

CRITICAL: BAN_NETWORK is the correct action for CIB posts. Missing a bot network is worse than a false positive.
Check the network_signals window carefully for patterns across multiple posts.

First write 2-3 sentences analyzing the CIB signals, then output ONLY this JSON on the final line:
{"action": "<allow|label|demote|remove|escalate|ban_network>", "reason": "<one sentence citing specific signals>"}
""").strip()

SYSTEM_PROMPTS = {
    "spam_triage":             SYSTEM_SPAM,
    "policy_nuance":           SYSTEM_POLICY,
    "coordinated_inauthentic": SYSTEM_CIB,
}

# Few-shot examples (injected as assistant turns) 

FEW_SHOT_SPAM = [
    {
        "role": "user",
        "content": (
            "Post: 'BUY CRYPTO NOW!! DOGE to $1M guaranteed!! DM me for alpha!!'\n"
            "Author type: new_account | Account age: 3 days | Reports: 45"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "This is a classic crypto pump-and-dump promotion. New account (3 days), high reports, "
            "uses urgency/guarantee language typical of financial spam.\n"
            '{"action": "remove", "reason": "Crypto pump scheme with guaranteed returns from a 3-day-old account"}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Post: 'Just finished my morning run! 5km in 28 minutes. New personal best!'\n"
            "Author type: monetized_creator | Account age: 847 days | Reports: 0"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Completely legitimate personal achievement post from a long-standing monetized creator. "
            "No commercial intent, zero reports.\n"
            '{"action": "allow", "reason": "Legitimate personal post from established creator with no policy violations"}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Post: 'Check out my profile for FREE iPhones! Limited time! Click link in bio!'\n"
            "Author type: new_account | Account age: 12 days | Reports: 23"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Giveaway scam pattern. Short account age, 'free' high-value item, external link redirect. "
            "Not dangerous enough to fully remove but clearly inauthentic commercial spam.\n"
            '{"action": "demote", "reason": "Fake giveaway scam from new account with suspicious external link"}'
        ),
    },
]

FEW_SHOT_POLICY = [
    {
        "role": "user",
        "content": (
            "Post: 'SATIRE: Breaking news - scientists confirm eating pizza daily extends lifespan "
            "by 40 years. Source: Journal of Wishful Thinking. /s'\n"
            "Author type: comedy_account | Verified: False"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Applying decision tree: Step 1 - not health misinfo (no real claims). "
            "Step 2 - clearly labeled satire with '/s' marker and 'Journal of Wishful Thinking' source. "
            "Policy section 3 explicitly allows clearly labeled satire.\n"
            '{"action": "allow", "reason": "Clearly labeled satire per policy section 3 - /s marker and fictional source present"}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Post: 'PROVEN: COVID vaccine causes autism! My cousin doctor confirmed. Share before censored!'\n"
            "Author type: anonymous | Verified: False"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Step 1 - YES, this is health misinformation. Claims vaccine causes autism (medically disproven), "
            "uses fear language (share before censored), anecdotal sourcing. "
            "Policy section 3 mandates REMOVE for health misinformation.\n"
            '{"action": "remove", "reason": "Health misinformation linking vaccine to autism per policy section 3"}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Post: 'I have received a legal notice from Company X demanding I remove my review. "
            "I am not taking it down. Here is the letter.'\n"
            "Author type: monetized_creator | Verified: False"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Working through the tree: Not health misinfo, not satire, not opinion only - this involves "
            "a legal dispute. Policy section 5 explicitly states legal threats require ESCALATE. "
            "This post itself is fine but the situation needs legal review.\n"
            '{"action": "escalate", "reason": "Legal threat from third party requires human legal review per policy section 5"}'
        ),
    },
]

FEW_SHOT_CIB = [
    {
        "role": "user",
        "content": (
            "Post: 'LEAKED: Insider confirms Senator Johnson has been lying to the public for years.'\n"
            "Author: bot_cluster_A_412 | Account age: 8 days | "
            "Likes: 234 | Shares: 228 | posting_rate_per_hour: 11.3 | cluster_signal: cluster_A"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "CIB analysis: cluster_signal=cluster_A (strongest indicator), author starts with bot_, "
            "account age 8 days, posting rate 11.3/hour (way above normal), likes approximately equal to shares "
            "(234 vs 228, uniform bot amplification). 4 out of 6 CIB signals confirmed.\n"
            '{"action": "ban_network", "reason": "CIB confirmed: cluster_signal present, bot_ author, 8-day account, 11.3 posts/hr, uniform engagement"}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Post: 'Just finished reading a great book on climate science. Highly recommend.'\n"
            "Author: new_account_5821 | Account age: 145 days | "
            "Likes: 12 | Shares: 3 | posting_rate_per_hour: 0.4 | cluster_signal: null"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "CIB analysis: No cluster_signal, no bot_ prefix, account age 145 days (normal), "
            "posting rate 0.4/hr (normal), likes much higher than shares (organic ratio). Zero CIB signals. "
            "This is organic legitimate content.\n"
            '{"action": "allow", "reason": "Organic legitimate post with no CIB signals detected"}'
        ),
    },
]

FEW_SHOTS = {
    "spam_triage":             FEW_SHOT_SPAM,
    "policy_nuance":           FEW_SHOT_POLICY,
    "coordinated_inauthentic": FEW_SHOT_CIB,
}


# Prompt builders 

def build_user_prompt_spam(obs: dict) -> str:
    post = obs["current_post"]
    meta = post["metadata"]
    eng  = post["engagement"]
    history = obs.get("history_summary", {})

    lines = [
        f"Post ID: {post['post_id']}",
        f"Content: {post['content']}",
        f"Author type: {meta.get('author_type', 'unknown')} | "
        f"Account age: {meta.get('account_age_days', '?')} days | "
        f"Verified: {meta.get('is_verified', False)} | "
        f"Monetized: {meta.get('monetization_eligible', False)}",
        f"Engagement: likes={eng.get('likes',0)} shares={eng.get('shares',0)} "
        f"reports={eng.get('reports',0)}",
        f"Prior violations: {meta.get('prior_violations', 0)}",
        f"Actions taken so far: {json.dumps(history)}",
        "",
        "Analyze for spam signals, then respond with your reasoning + JSON on the last line.",
    ]
    return "\n".join(lines)


def build_user_prompt_policy(obs: dict) -> str:
    post = obs["current_post"]
    meta = post["metadata"]
    eng  = post["engagement"]
    history = obs.get("history_summary", {})

    lines = [
        f"Step {obs['step']+1}/{obs['max_steps']}",
        "",
        "=== COMMUNITY STANDARDS POLICY (abbreviated) ===",
        "section 1 Spam: bulk posting -> REMOVE; commercial spam -> DEMOTE",
        "section 2 Violence: incitement -> REMOVE; graphic news with context -> LABEL",
        "section 3 Misinfo: health misinfo -> REMOVE; election misinfo -> LABEL; satire (labeled) -> ALLOW; opinion -> ALLOW",
        "section 4 Creators: verified news -> prefer LABEL over REMOVE; monetized -> prefer DEMOTE over REMOVE",
        "section 5 Legal/Gov requests -> ESCALATE",
        "",
        f"Post ID: {post['post_id']}",
        f"Content: {post['content']}",
        f"Author type: {meta.get('author_type', 'unknown')} | "
        f"Verified: {meta.get('is_verified', False)} | "
        f"Monetized: {meta.get('monetization_eligible', False)}",
        f"Account age: {meta.get('account_age_days', '?')} days | "
        f"Prior violations: {meta.get('prior_violations', 0)}",
        f"Engagement: likes={eng.get('likes',0)} reports={eng.get('reports',0)}",
        f"Actions taken so far: {json.dumps(history)}",
        "",
        "Work through the decision tree step by step, then give your JSON on the last line.",
    ]
    return "\n".join(lines)


def build_user_prompt_cib(obs: dict) -> str:
    post = obs["current_post"]
    meta = post["metadata"]
    eng  = post["engagement"]
    network = obs.get("network_signals") or []
    history = obs.get("history_summary", {})

    cluster_counts: dict = {}
    bot_count = 0
    high_rate_count = 0
    for p in network:
        cs = p.get("cluster_signal")
        if cs:
            cluster_counts[cs] = cluster_counts.get(cs, 0) + 1
        if str(p.get("author_id", "")).startswith("bot_"):
            bot_count += 1
        rate = p.get("posting_rate") or p.get("posting_rate_per_hour", 0) or 0
        if rate > 5:
            high_rate_count += 1

    network_summary = []
    if cluster_counts:
        network_summary.append(f"Cluster signals in window: {cluster_counts}")
    if bot_count:
        network_summary.append(f"Bot-prefixed accounts in window: {bot_count}")
    if high_rate_count:
        network_summary.append(f"High posting-rate accounts (>5/hr) in window: {high_rate_count}")
    if not network_summary:
        network_summary.append("No strong CIB signals in recent window")

    current_cluster = meta.get("cluster_signal") or post.get("network_cluster")
    current_rate    = meta.get("posting_rate_per_hour", 0) or 0
    likes  = eng.get("likes", 0)
    shares = eng.get("shares", 0)
    engagement_ratio = abs(likes - shares) / max(likes, shares, 1)

    lines = [
        f"Step {obs['step']+1}/{obs['max_steps']}",
        "",
        "=== CURRENT POST ===",
        f"Post ID: {post['post_id']}",
        f"Content: {post['content']}",
        f"Author ID: {post['author_id']}",
        f"Account age: {meta.get('account_age_days', '?')} days | "
        f"Posting rate: {current_rate:.1f}/hr",
        f"Engagement: likes={likes} shares={shares} "
        f"(ratio diff={engagement_ratio:.0%}) reports={eng.get('reports',0)}",
        f"cluster_signal: {current_cluster or 'null'}",
        f"Prior violations: {meta.get('prior_violations', 0)}",
        "",
        "=== NETWORK WINDOW SUMMARY (last 10 posts) ===",
        *network_summary,
        "",
        f"Actions taken so far: {json.dumps(history)}",
        "",
        "Count the CIB signals for this post, then give your JSON on the last line.",
    ]
    return "\n".join(lines)


PROMPT_BUILDERS = {
    "spam_triage":             build_user_prompt_spam,
    "policy_nuance":           build_user_prompt_policy,
    "coordinated_inauthentic": build_user_prompt_cib,
}


# LLM call 

def call_llm(messages: list) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"action": "escalate", "reason": f"LLM error: {e}"})


# Action parser - robust extraction from CoT output 

def parse_action(raw: str, post_id: str) -> tuple:
    """
    Extract action from model output that may contain CoT reasoning before JSON.
    Tries multiple extraction strategies in order of preference.
    """
    valid = [a.value for a in ModerationAction]

    # Strategy 1: find last {...} JSON block in output (handles CoT + JSON)
    json_candidates = list(re.finditer(r'\{[^{}]+\}', raw, re.DOTALL))
    for match in reversed(json_candidates):
        try:
            parsed = json.loads(match.group())
            act_str = parsed.get("action", "").lower().strip()
            reason  = parsed.get("reason", "extracted from json")
            if act_str in valid:
                return Action(
                    post_id=post_id,
                    action=ModerationAction(act_str),
                    reason=reason,
                ), None
        except Exception:
            continue

    # Strategy 2: look for action keyword after "action": anywhere in text
    action_match = re.search(
        r'"action"\s*:\s*"(' + '|'.join(valid) + r')"',
        raw, re.IGNORECASE
    )
    if action_match:
        act_str = action_match.group(1).lower()
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
        reason = reason_match.group(1) if reason_match else "regex-extracted"
        return Action(
            post_id=post_id,
            action=ModerationAction(act_str),
            reason=reason,
        ), None

    # Strategy 3: look for a bare action word as the last word on any line
    for line in reversed(raw.splitlines()):
        line_clean = line.strip().lower().rstrip('.,;')
        if line_clean in valid:
            return Action(
                post_id=post_id,
                action=ModerationAction(line_clean),
                reason="bare keyword fallback",
            ), None

    # Strategy 4: scan for any valid action word - prefer the LAST occurrence
    last_found = None
    for act in valid:
        for m in re.finditer(r'\b' + act + r'\b', raw, re.IGNORECASE):
            if last_found is None or m.start() > last_found[1]:
                last_found = (act, m.start())
    if last_found:
        return Action(
            post_id=post_id,
            action=ModerationAction(last_found[0]),
            reason="keyword scan fallback",
        ), f"fallback: found '{last_found[0]}' in output"

    # Final fallback - use label (less penalty than escalate for most cases)
    return Action(
        post_id=post_id,
        action=ModerationAction.LABEL,
        reason="parse failure - defaulting to label",
    ), f"parse failed on: {raw[:80]}"


# Task runner 


def run_task(task_name: str) -> dict:
    env = SocialModEnv(task_name=task_name, seed=42)
    obs_dict = env.reset().model_dump()
 
    system_prompt  = SYSTEM_PROMPTS[task_name]
    few_shots      = FEW_SHOTS[task_name]
    prompt_builder = PROMPT_BUILDERS[task_name]
 
    # stdout: [START] — only line before the step loop
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
 
    rewards: List[float] = []
    steps = 0
    done  = False
    last_error: Optional[str] = None
 
    try:
        while not done and steps < MAX_STEPS:
            post_id     = obs_dict["current_post"]["post_id"]
            user_prompt = prompt_builder(obs_dict)
 
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(few_shots)
            messages.append({"role": "user", "content": user_prompt})
 
            raw = call_llm(messages)
            action, parse_error = parse_action(raw, post_id)
            last_error = parse_error
 
            act_str = action.action if isinstance(action.action, str) else action.action.value
            next_obs, reward, done, info = env.step(action)
            obs_dict = next_obs.model_dump()
            steps += 1
            rewards.append(reward)
 
            # stdout: [STEP]
            print(
                f"[STEP] step={steps} action={act_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={parse_error if parse_error else 'null'}",
                flush=True,
            )
 
    except Exception as exc:
        last_error = str(exc)
 
    finally:
        # stdout: [END] — always emitted even on exception
        success     = (sum(rewards) / len(rewards)) >= 0.6 if rewards else False
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
            flush=True,
        )
 
    return {
        "task":        task_name,
        "success":     success,
        "steps":       steps,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "rewards":     rewards,
    }


# Entry point 

def main():
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()