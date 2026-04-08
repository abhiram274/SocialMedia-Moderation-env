"""
SocialModEnv — Content Moderation Policy Simulation Environment
OpenEnv-compliant RL environment for training and evaluating
AI agents on real-world content moderation policy decisions.
"""

from __future__ import annotations

import json
import random
import re
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Utility Functions
MIN_SCORE = 1e-6
MAX_SCORE = 1 - 1e-6
def clamp_score(score: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, score))

# Action / Observation / Reward Models

class ModerationAction(str, Enum):
    ALLOW         = "allow"
    LABEL         = "label"          # add warning label
    DEMOTE        = "demote"         # reduce reach
    REMOVE        = "remove"         # take down entirely
    ESCALATE      = "escalate"       # send to human review
    BAN_NETWORK   = "ban_network"    # Task 3: network-level action


class Action(BaseModel):
    post_id: str
    action: ModerationAction
    reason: Optional[str] = Field(None, description="Agent's stated justification")

    class Config:
        use_enum_values = True


class Post(BaseModel):
    post_id: str
    content: str
    author_id: str
    engagement: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ground_truth: Optional[str] = Field(None, description="Hidden label for grading")
    network_cluster: Optional[str] = Field(None, description="For Task 3 only")


class Observation(BaseModel):
    step: int
    current_post: Post
    policy_document: str
    history_summary: Dict[str, int]   # running action counts
    network_signals: Optional[List[Dict]] = None  # Task 3 window
    task_name: str
    max_steps: int

    class Config:
        arbitrary_types_allowed = True


class Reward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    safety_score: float
    expression_score: float
    creator_score: float
    consistency_penalty: float
    breakdown: Dict[str, float] = Field(default_factory=dict)

# Synthetic Data Generator

POLICY_DOC = """
COMMUNITY STANDARDS POLICY v2.1

1. SPAM & INAUTHENTIC BEHAVIOR
   - Automated/bulk posting without disclosure → REMOVE
   - Coordinated inauthentic networks → BAN_NETWORK
   - Unsolicited commercial content → DEMOTE

2. HARMFUL CONTENT
   - Incitement to real-world violence → REMOVE
   - Graphic violence without news context → REMOVE
   - Hate speech targeting protected groups → REMOVE
   - Mild insults not targeting groups → LABEL

3. MISINFORMATION
   - Health misinformation (vaccines, medicine) → REMOVE
   - Political misinformation (election fraud claims without evidence) → LABEL + DEMOTE
   - Satire clearly labeled as such → ALLOW
   - Opinion and commentary → ALLOW

4. CREATOR PROTECTIONS
   - Verified news organizations → lower removal threshold, prefer LABEL
   - Monetized creators: prefer DEMOTE over REMOVE when borderline
   - Appeals process available for all REMOVE actions

5. ESCALATION
   - Novel content types not covered above → ESCALATE
   - Legal threats or government takedown requests → ESCALATE
   - Content involving minors in ambiguous context → ESCALATE
"""


class SyntheticPostGenerator:
    """Generates realistic synthetic posts with hidden ground truth labels."""

    SPAM_TEMPLATES = [
        ("BUY CRYPTO NOW!! 🚀 {coin} to $1M guaranteed!! DM me for alpha!!",
         "remove", "monetized_creator"),
        ("Check out my profile for FREE {item}! Limited time! Click link in bio!",
         "demote", "new_account"),
        ("URGENT: Your account will be suspended unless you verify at {url}",
         "remove", "bot"),
        ("I made $5000 working from home with this ONE trick. PM me.",
         "demote", "monetized_creator"),
        ("Follow for follow! Like4like! Grow your account fast!",
         "demote", "new_account"),
    ]

    HARMFUL_TEMPLATES = [
        ("We should round up all {group} and teach them a lesson they won't forget.",
         "remove", "anonymous"),
        ("Breaking: Video shows police brutality at {city} protest. Watch before deleted.",
         "label", "news_org"),
        ("I hate {group}, they ruin everything. Someone should do something.",
         "remove", "anonymous"),
        ("Graphic: Medical procedure footage from {hospital} — viewer discretion advised.",
         "label", "news_org"),
    ]

    MISINFO_TEMPLATES = [
        ("PROVEN: {vaccine} causes {disease}! My cousin's doctor confirmed. Share before censored!",
         "remove", "anonymous"),
        ("Satire: Scientists discover {food} cures all disease. /s — this is a joke btw.",
         "allow", "comedy_account"),
        ("BREAKING: Election in {state} was stolen! 10,000 fake ballots found (no source)",
         "label", "anonymous"),
        ("Opinion: I believe the {policy} bill is bad for small businesses. Here's why...",
         "allow", "monetized_creator"),
        ("Study shows {supplement} reduces cancer risk — link to actual NEJM paper attached.",
         "allow", "news_org"),
    ]

    LEGIT_TEMPLATES = [
        ("Just finished my morning run! 5km in 28 minutes. New personal best!",
         "allow", "monetized_creator"),
        ("Reminder: Community cleanup event this Saturday at {park}. Bring gloves!",
         "allow", "community_org"),
        ("Hot take: {show} season 3 is the best one. Change my mind.",
         "allow", "new_account"),
        ("Grateful for everyone who supported my small business this year. Thank you.",
         "allow", "monetized_creator"),
        ("Thread: Everything wrong with the new {app} update (1/12).",
         "allow", "monetized_creator"),
    ]

    # For Task 3 — coordinated inauthentic behavior
    CIB_CLUSTER_TEMPLATES = [
        "New investigation proves {target} is corrupt. Share to expose the truth!",
        "LEAKED: Insider confirms {target} has been lying to the public for years.",
        "{target} must resign NOW. The evidence is overwhelming. RT to spread!",
        "I used to support {target} but after seeing this I'm done. Wake up people!",
        "Don't let the media bury this. {target} scandal is real. #TruthMatters",
    ]

    FILLERS = {
        "coin": ["DOGE", "SHIB", "PEPE", "LUNA"],
        "item": ["iPhone", "PS5", "Airpods", "Gift cards"],
        "url": ["verify-account.xyz", "login-secure.net"],
        "group": ["immigrants", "conservatives", "liberals", "vegans"],
        "city": ["Portland", "Atlanta", "Minneapolis", "Chicago"],
        "hospital": ["Stanford Medical", "Johns Hopkins", "Mayo Clinic"],
        "vaccine": ["COVID vaccine", "flu shot", "MMR vaccine"],
        "disease": ["autism", "infertility", "microchips", "5G activation"],
        "food": ["broccoli", "turmeric", "celery juice", "coffee"],
        "state": ["Arizona", "Georgia", "Wisconsin", "Pennsylvania"],
        "policy": ["infrastructure", "climate", "tax", "healthcare"],
        "supplement": ["Vitamin D", "Omega-3", "Magnesium"],
        "show": ["Succession", "The Bear", "Severance", "White Lotus"],
        "app": ["Instagram", "TikTok", "Twitter", "Facebook"],
        "park": ["Riverside Park", "Central Park", "Green Lake"],
        "target": ["Senator Johnson", "CEO Martinez", "Director Williams"],
    }

    def _fill(self, template: str) -> str:
        def replacer(m):
            key = m.group(1)
            return random.choice(self.FILLERS.get(key, [key]))
        return re.sub(r"\{(\w+)\}", replacer, template)

    def generate_post(self, post_id: str, category: str = "mixed",
                      network_cluster: Optional[str] = None) -> Post:
        pools = {
            "spam": self.SPAM_TEMPLATES,
            "harmful": self.HARMFUL_TEMPLATES,
            "misinfo": self.MISINFO_TEMPLATES,
            "legit": self.LEGIT_TEMPLATES,
            "mixed": (self.SPAM_TEMPLATES + self.HARMFUL_TEMPLATES +
                      self.MISINFO_TEMPLATES + self.LEGIT_TEMPLATES),
        }
        template, ground_truth, author_type = random.choice(pools.get(category, pools["mixed"]))
        content = self._fill(template)

        return Post(
            post_id=post_id,
            content=content,
            author_id=f"{author_type}_{random.randint(1000,9999)}",
            engagement={
                "likes": random.randint(0, 50000),
                "shares": random.randint(0, 10000),
                "comments": random.randint(0, 5000),
                "reports": random.randint(0, 200),
            },
            metadata={
                "author_type": author_type,
                "account_age_days": random.randint(1, 3000),
                "is_verified": author_type == "news_org",
                "monetization_eligible": author_type == "monetized_creator",
                "prior_violations": random.randint(0, 5),
                "timestamp": time.time(),
            },
            ground_truth=ground_truth,
            network_cluster=network_cluster,
        )

    def generate_cib_post(self, post_id: str, cluster_id: str) -> Post:
        """Generate a post that's part of a coordinated inauthentic behavior cluster."""
        template = random.choice(self.CIB_CLUSTER_TEMPLATES)
        content = self._fill(template)
        return Post(
            post_id=post_id,
            content=content,
            author_id=f"bot_{cluster_id}_{random.randint(100,999)}",
            engagement={
                "likes": random.randint(50, 500),    # suspiciously uniform
                "shares": random.randint(40, 480),
                "comments": random.randint(5, 50),
                "reports": random.randint(0, 10),
            },
            metadata={
                "author_type": "new_account",
                "account_age_days": random.randint(1, 30),
                "is_verified": False,
                "monetization_eligible": False,
                "prior_violations": 0,
                "posting_rate_per_hour": random.uniform(8, 15),   # high rate
                "timestamp": time.time(),
                "cluster_signal": cluster_id,  # visible to agent
            },
            ground_truth="ban_network",
            network_cluster=cluster_id,
        )


# Task Definitions

class Task1SpamTriage:
    """
    EASY: Binary spam classification.
    Agent receives a stream of clearly-spam or clearly-legitimate posts.
    Grader: precision/recall against ground truth.
    Expected score: 0.75–0.90 for frontier models.
    """
    name = "spam_triage"
    description = "Classify spam vs. legitimate content. Remove or demote spam, allow legit."
    max_steps = 20

    def __init__(self):
        self.gen = SyntheticPostGenerator()
        self.posts: List[Post] = []
        self.correct_actions = {"remove", "demote"}
        self.legit_actions = {"allow"}

    def generate_posts(self) -> List[Post]:
        posts = []
        for i in range(10):
            posts.append(self.gen.generate_post(f"t1_spam_{i}", category="spam"))
        for i in range(10):
            posts.append(self.gen.generate_post(f"t1_legit_{i}", category="legit"))
        random.shuffle(posts)
        return posts

    def grade_action(self, post: Post, action: ModerationAction) -> Tuple[float, str]:
        gt = post.ground_truth
        a = action.value if hasattr(action, 'value') else action

        if gt in ("remove", "demote"):
            if a in ("remove", "demote"):
                return clamp_score(1.0), "correct: spam actioned"
            elif a == "escalate":
                return clamp_score(0.5), "partial: escalated instead of actioning"
            else:
                return clamp_score(0.0), "incorrect: spam allowed"
        elif gt == "allow":
            if a == "allow":
                return clamp_score(1.0), "correct: legit content allowed"
            elif a == "label":
                return clamp_score(0.7), "partial: unnecessary label on clean content"
            elif a == "demote":
                return clamp_score(0.3), "over-removal: legitimate post demoted"
            else:
                return clamp_score(0.0), "over-removal: legitimate post removed"
        return clamp_score(0.5), "unclear case"


class Task2PolicyNuance:
    """
    MEDIUM: Gray-zone moderation requiring policy document reading.
    Posts live near category boundaries — satire, news imagery, political speech.
    Grader: accuracy against expert annotations, with partial credit.
    Expected score: 0.45–0.70 for frontier models.
    """
    name = "policy_nuance"
    description = (
        "Apply community standards policy to borderline posts. "
        "Read the policy document carefully. Partial credit for near-misses."
    )
    max_steps = 20

    NUANCE_POSTS = [
        {
            "content": "SATIRE: Breaking news — scientists confirm that eating pizza daily "
                       "extends lifespan by 40 years. Source: Journal of Wishful Thinking. /s",
            "author_type": "comedy_account",
            "ground_truth": "allow",
            "gold_actions": ["allow"],
            "partial_actions": ["label"],
            "wrong_actions": ["remove", "demote"],
            "note": "Clearly labeled satire → ALLOW per policy §3",
        },
        {
            "content": "My political opinion: the new infrastructure bill will hurt rural communities "
                       "because it prioritizes urban transit over road maintenance. Here's my analysis.",
            "author_type": "monetized_creator",
            "ground_truth": "allow",
            "gold_actions": ["allow"],
            "partial_actions": ["label"],
            "wrong_actions": ["remove", "demote"],
            "note": "Opinion/commentary → ALLOW per policy §3",
        },
        {
            "content": "LEAKED FOOTAGE: What the government doesn't want you to see about the "
                       "election results in Maricopa County. 50,000 ballots switched overnight!!!",
            "author_type": "anonymous",
            "ground_truth": "label",
            "gold_actions": ["label", "demote"],
            "partial_actions": ["escalate"],
            "wrong_actions": ["allow", "remove"],
            "note": "Election misinfo → LABEL + DEMOTE per policy §3",
        },
        {
            "content": "Warning: graphic medical content. This video from the OR shows a real "
                       "appendectomy procedure. Educational content for medical students.",
            "author_type": "news_org",
            "ground_truth": "label",
            "gold_actions": ["label"],
            "partial_actions": ["demote", "allow"],
            "wrong_actions": ["remove"],
            "note": "Medical content from verified org → LABEL per policy §2",
        },
        {
            "content": "I personally think conservatives are stupid and should be banned from "
                       "public life. They ruin everything they touch.",
            "author_type": "anonymous",
            "ground_truth": "label",
            "gold_actions": ["label", "demote"],
            "partial_actions": ["remove", "escalate"],
            "wrong_actions": ["allow"],
            "note": "Mild political insult not targeting protected class → LABEL",
        },
        {
            "content": "Study from Harvard Medical School shows Vitamin D supplementation "
                       "reduces colorectal cancer risk by 17%. Link to NEJM paper in bio.",
            "author_type": "news_org",
            "ground_truth": "allow",
            "gold_actions": ["allow"],
            "partial_actions": ["label"],
            "wrong_actions": ["remove", "demote"],
            "note": "Legitimate health content with credible source → ALLOW",
        },
        {
            "content": "The government is putting chemicals in the water to sterilize us. "
                       "The flu vaccine causes autism. SHARE BEFORE DELETED.",
            "author_type": "anonymous",
            "ground_truth": "remove",
            "gold_actions": ["remove"],
            "partial_actions": ["label", "demote"],
            "wrong_actions": ["allow", "escalate"],
            "note": "Health misinformation → REMOVE per policy §3",
        },
        {
            "content": "My small business made a little girl cry because I couldn't fulfill "
                       "her birthday cake order. I feel terrible. Here's how I made it right.",
            "author_type": "monetized_creator",
            "ground_truth": "allow",
            "gold_actions": ["allow"],
            "partial_actions": [],
            "wrong_actions": ["label", "demote", "remove"],
            "note": "Wholesome creator content → ALLOW",
        },
        {
            "content": "Breaking: Video shows group of men attacking a homeless man in "
                       "downtown Seattle. Authorities are investigating. Warning: disturbing.",
            "author_type": "news_org",
            "ground_truth": "label",
            "gold_actions": ["label"],
            "partial_actions": ["demote", "escalate"],
            "wrong_actions": ["remove", "allow"],
            "note": "News violence with context → LABEL per policy §2",
        },
        {
            "content": "I have received a legal notice from [Company X] demanding I remove "
                       "my review of their product. I'm not taking it down. Here's the letter.",
            "author_type": "monetized_creator",
            "ground_truth": "escalate",
            "gold_actions": ["escalate"],
            "partial_actions": ["allow"],
            "wrong_actions": ["remove", "label", "demote"],
            "note": "Legal threat → ESCALATE per policy §5",
        },
    ] * 2  # 20 posts total

    def __init__(self):
        self.posts: List[Dict] = []

    def generate_posts(self) -> List[Post]:
        pool = deepcopy(self.NUANCE_POSTS)
        random.shuffle(pool)
        posts = []
        for i, p in enumerate(pool[:20]):
            posts.append(Post(
                post_id=f"t2_{i}",
                content=p["content"],
                author_id=f"{p['author_type']}_{random.randint(1000,9999)}",
                engagement={
                    "likes": random.randint(100, 10000),
                    "shares": random.randint(50, 5000),
                    "comments": random.randint(10, 1000),
                    "reports": random.randint(0, 100),
                },
                metadata={
                    "author_type": p["author_type"],
                    "account_age_days": random.randint(30, 2000),
                    "is_verified": p["author_type"] == "news_org",
                    "monetization_eligible": p["author_type"] == "monetized_creator",
                    "prior_violations": 0,
                    "note": p["note"],
                    "timestamp": time.time(),
                },
                ground_truth=p["ground_truth"],
            ))
            posts[-1]._policy = p  # type: ignore[attr-defined]
        return posts

    def grade_action(self, post: Post, action: ModerationAction) -> Tuple[float, str]:
        a = action.value if hasattr(action, 'value') else action
        # Find matching policy definition
        for p in self.NUANCE_POSTS:
            if p["content"].strip() == post.content.strip():
                if a in p["gold_actions"]:
                    return clamp_score(1.0), f"correct: {p['note']}"
                elif a in p["partial_actions"]:
                    return clamp_score(0.5), f"partial credit: near-miss. {p['note']}"
                else:
                    return clamp_score(0.0), f"incorrect. {p['note']}"
        # Fallback
        if a == post.ground_truth:
            return clamp_score(1.0), "correct"
        return clamp_score(0.0), "incorrect"


class Task3CoordinatedInauthentic:
    """
    HARD: Detect and action coordinated inauthentic behavior (CIB) networks.
    Individual posts may look benign. Agent must identify the network pattern
    across a 100-post stream and use BAN_NETWORK action.
    Grader: network-level detection accuracy + individual post decisions.
    Expected score: 0.25–0.45 for frontier models.
    """
    name = "coordinated_inauthentic"
    description = (
        "Detect coordinated inauthentic behavior networks. Individual posts "
        "may seem borderline. Look for network-level patterns: "
        "uniform engagement, account age, posting rate, shared narratives. "
        "Use BAN_NETWORK to action an entire cluster."
    )
    max_steps = 30

    def __init__(self):
        self.gen = SyntheticPostGenerator()
        self.cluster_ids = ["cluster_A", "cluster_B"]
        self.organic_ratio = 0.5

    def generate_posts(self) -> List[Post]:
        posts = []
        # Mix organic and CIB posts
        n_per_cluster = 7
        n_organic = 16

        for cluster in self.cluster_ids:
            for i in range(n_per_cluster):
                posts.append(self.gen.generate_cib_post(
                    f"t3_{cluster}_{i}", cluster
                ))

        for i in range(n_organic):
            posts.append(self.gen.generate_post(
                f"t3_organic_{i}",
                category=random.choice(["legit", "misinfo", "spam"])
            ))

        random.shuffle(posts)
        return posts[:30]

    def grade_action(self, post: Post, action: ModerationAction,
                     action_history: List[Action]) -> Tuple[float, str]:
        a = action.value if hasattr(action, 'value') else action

        if post.network_cluster:
            # This is a CIB post
            if a == "ban_network":
                return clamp_score(1.0), "correct: network-level action on CIB post"
            elif a == "remove":
                return clamp_score(0.4), "partial: removed post but missed network pattern"
            elif a in ("demote", "label"):
                return clamp_score(0.2), "partial: weak action on CIB post"
            elif a == "escalate":
                return clamp_score(0.5), "partial: escalated — appropriate but BAN_NETWORK preferred"
            else:
                return clamp_score(0.0), "incorrect: CIB post allowed"
        else:
            # Organic post — check if agent is trigger-happy with BAN_NETWORK
            gt = post.ground_truth or "allow"
            if a == "ban_network":
                return clamp_score(0.0), "false positive: BAN_NETWORK on organic content"
            if a == gt:
                return clamp_score(1.0), "correct: organic post handled correctly"
            elif a in ("label", "demote") and gt == "allow":
                return clamp_score(0.4), "over-cautious but not catastrophic"
            elif a == "allow" and gt in ("remove", "demote"):
                return clamp_score(0.0), "missed violation"
            elif a == "remove" and gt == "allow":
                return clamp_score(0.0), "false positive: legitimate content removed"
            return clamp_score(0.5), "borderline case"


# ─────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────

class SocialModEnv:
    """
    OpenEnv-compliant Content Moderation Policy Simulation Environment.
    Supports 3 tasks of increasing difficulty.
    """

    TASKS = {
        "spam_triage": Task1SpamTriage,
        "policy_nuance": Task2PolicyNuance,
        "coordinated_inauthentic": Task3CoordinatedInauthentic,
    }

    def __init__(self, task_name: str = "spam_triage", seed: int = 42):
        if task_name not in self.TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(self.TASKS)}")

        self.task_name = task_name
        self.seed = seed
        random.seed(seed)

        self.task = self.TASKS[task_name]()
        self._posts: List[Post] = []
        self._step_idx: int = 0
        self._action_history: List[Action] = []
        self._rewards_per_step: List[float] = []
        self._action_counts: Dict[str, int] = {}
        self._episode_scores: List[float] = []
        self._done: bool = False
        self._initialized: bool = False

    def reset(self) -> Observation:
        random.seed(self.seed)
        self._posts = self.task.generate_posts()
        self._step_idx = 0
        self._action_history = []
        self._rewards_per_step = []
        self._episode_scores = []
        self._action_counts = {a.value: 0 for a in ModerationAction}
        self._done = False
        self._initialized = True
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode done. Call reset().")

        current_post = self._posts[self._step_idx]

        # Grade the action
        reward_val, reason = self._grade(current_post, action)

        # Consistency penalty — same post_type getting wildly different actions
        consistency_penalty = self._compute_consistency_penalty(action)

        # Shaped reward
        shaped = self._shape_reward(
            base=reward_val,
            post=current_post,
            action=action,
            consistency_penalty=consistency_penalty,
        )

        self._action_history.append(action)
        act_val = action.action if isinstance(action.action, str) else action.action.value
        self._action_counts[act_val] = self._action_counts.get(act_val, 0) + 1
        self._episode_scores.append(shaped)
        self._rewards_per_step.append(shaped)
        self._step_idx += 1
        self._done = self._step_idx >= min(len(self._posts), self.task.max_steps)

        obs = self._make_observation() if not self._done else self._make_observation()

        info = {
            "reason": reason,
            "consistency_penalty": consistency_penalty,
            "ground_truth": current_post.ground_truth,
            "post_id": current_post.post_id,
            "episode_mean_reward": clamp_score(sum(self._episode_scores) / len(self._episode_scores)),
        }
        shaped = clamp_score(shaped)
        return obs, shaped, self._done, info

    def state(self) -> Dict[str, Any]:
        mean_reward = (
            sum(self._episode_scores) / len(self._episode_scores)
            if self._episode_scores else MIN_SCORE
        )
        mean_reward = clamp_score(mean_reward)
        return {
            "task_name": self.task_name,
            "step": self._step_idx,
            "max_steps": self.task.max_steps,
            "done": self._done,
            "action_history": [a.dict() for a in self._action_history],
            "action_counts": self._action_counts,
            "rewards": self._rewards_per_step,
            "episode_mean_reward": mean_reward,
        }

    # ── Internal helpers ──────────────────────────────────────

    def _make_observation(self) -> Observation:
        if self._step_idx >= len(self._posts):
            # terminal observation — return last post
            current_post = self._posts[-1]
        else:
            current_post = self._posts[self._step_idx]

        # For Task 3: provide a window of recent posts as network signals
        network_signals = None
        if self.task_name == "coordinated_inauthentic" and self._step_idx > 0:
            window = self._posts[max(0, self._step_idx - 10):self._step_idx]
            network_signals = [
                {
                    "post_id": p.post_id,
                    "author_id": p.author_id,
                    "engagement": p.engagement,
                    "account_age_days": p.metadata.get("account_age_days"),
                    "posting_rate": p.metadata.get("posting_rate_per_hour"),
                    "cluster_signal": p.metadata.get("cluster_signal"),
                }
                for p in window
            ]

        return Observation(
            step=self._step_idx,
            current_post=current_post,
            policy_document=POLICY_DOC,
            history_summary=dict(self._action_counts),
            network_signals=network_signals,
            task_name=self.task_name,
            max_steps=self.task.max_steps,
        )

    def _grade(self, post: Post, action: Action) -> Tuple[float, str]:
        act = action.action
        if self.task_name == "spam_triage":
            return self.task.grade_action(post, act)
        elif self.task_name == "policy_nuance":
            return self.task.grade_action(post, act)
        elif self.task_name == "coordinated_inauthentic":
            return self.task.grade_action(post, act, self._action_history)
        return clamp_score(0.0), "unknown task"

    def _compute_consistency_penalty(self, action: Action) -> float:
        """
        Penalize inconsistent policy application.
        If the same author_type was actioned differently before, apply a small penalty.
        """
        if len(self._action_history) < 3:
            return 0.0  # not enough history for consistency check

        current_post = self._posts[self._step_idx]
        author_type = current_post.metadata.get("author_type", "unknown")

        # Find past actions on posts from same author_type
        past_actions_for_type = []
        for i, past_action in enumerate(self._action_history):
            if i < len(self._posts):
                past_post = self._posts[i]
                if (past_post.metadata.get("author_type") == author_type and
                        past_post.ground_truth == current_post.ground_truth):
                    past_act = past_action.action
                    if isinstance(past_act, ModerationAction):
                        past_act = past_act.value
                    past_actions_for_type.append(past_act)

        if not past_actions_for_type:
            return 0.0

        current_act = action.action
        if isinstance(current_act, ModerationAction):
            current_act = current_act.value

        # If current action differs from past actions for same post type → small penalty
        if current_act not in past_actions_for_type:
            return 0.05

        return 0.0

    def _shape_reward(self, base: float, post: Post, action: Action,
                      consistency_penalty: float) -> float:
        """
        Multi-objective reward shaping.
        Balances safety, expression rights, creator impact, and consistency.
        Penalizes corner solutions (always remove or always allow).
        """
        act = action.action
        if isinstance(act, ModerationAction):
            act = act.value

        # Safety bonus: removing/demoting genuinely harmful content
        safety_bonus = 0.0
        if post.ground_truth in ("remove", "demote", "ban_network") and act in ("remove", "ban_network"):
            safety_bonus = 0.05

        # Expression penalty: over-removing legitimate content
        expression_penalty = 0.0
        if post.ground_truth == "allow" and act == "remove":
            expression_penalty = 0.15

        # Creator penalty: removing monetized creator content without cause
        creator_penalty = 0.0
        if (post.metadata.get("monetization_eligible") and
                post.ground_truth in ("allow", "label") and
                act == "remove"):
            creator_penalty = 0.10

        # Escalation credit: appropriate uncertainty acknowledgment
        escalation_bonus = 0.0
        if act == "escalate" and post.ground_truth == "escalate":
            escalation_bonus = 0.05

        # Lazy policy penalty: if agent always does the same action
        lazy_penalty = 0.0
        if len(self._action_history) > 5:
            most_common = max(self._action_counts.values()) if self._action_counts else 0
            if most_common / max(len(self._action_history), 1) > 0.85:
                lazy_penalty = 0.08

        shaped = (
            base
            + safety_bonus
            + escalation_bonus
            - expression_penalty
            - creator_penalty
            - consistency_penalty
            - lazy_penalty
        )

        shaped = clamp_score(shaped)
        return float(f"{shaped:.6f}")