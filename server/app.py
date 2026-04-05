"""
SocialModEnv — FastAPI server exposing the OpenEnv HTTP interface.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.social_mod_env import (
    Action,
    ModerationAction,
    Observation,
    SocialModEnv,
)

app = FastAPI(
    title="SocialModEnv",
    description="Content Moderation Policy RL Environment — OpenEnv Spec",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task
_envs: Dict[str, SocialModEnv] = {}


def _get_env(task_name: str) -> SocialModEnv:
    if task_name not in _envs:
        _envs[task_name] = SocialModEnv(task_name=task_name)
    return _envs[task_name]


# Request / Response schemas 

class ResetRequest(BaseModel):
    task_name: str = "spam_triage"
    seed: int = 42


class StepRequest(BaseModel):
    task_name: str = "spam_triage"
    post_id: str
    action: str
    reason: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# Routes 

@app.get("/health")
def health():
    return {"status": "ok", "env": "SocialModEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "spam_triage",
                "difficulty": "easy",
                "max_steps": 20,
                "description": "Binary spam vs legitimate content classification",
                "valid_actions": ["allow", "label", "demote", "remove", "escalate"],
            },
            {
                "name": "policy_nuance",
                "difficulty": "medium",
                "max_steps": 20,
                "description": "Gray-zone moderation with policy document context",
                "valid_actions": ["allow", "label", "demote", "remove", "escalate"],
            },
            {
                "name": "coordinated_inauthentic",
                "difficulty": "hard",
                "max_steps": 30,
                "description": "Detect coordinated inauthentic behavior networks",
                "valid_actions": ["allow", "label", "demote", "remove", "escalate", "ban_network"],
            },
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest):
    env = SocialModEnv(task_name=req.task_name, seed=req.seed)
    _envs[req.task_name] = env
    obs = env.reset()
    return obs.dict()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if req.task_name not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{req.task_name}' not initialized. Call /reset first.",
        )

    valid_actions = [a.value for a in ModerationAction]
    if req.action not in valid_actions:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{req.action}'. Valid: {valid_actions}",
        )

    env = _envs[req.task_name]
    action = Action(
        post_id=req.post_id,
        action=ModerationAction(req.action),
        reason=req.reason,
    )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=obs.dict(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state(task_name: str = "spam_triage"):
    if task_name not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task_name}' not initialized. Call /reset first.",
        )
    return _envs[task_name].state()


# Entry point for local running
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()