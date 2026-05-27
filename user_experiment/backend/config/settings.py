"""Configuration management with environment variables.

Per user's CLAUDE.md requirements:
- NO default values or fallbacks
- Errors must surface immediately
"""

import os
from typing import Optional
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()


class SessionConfig(BaseModel):
    """Session configuration from environment."""

    deadline_time: int
    deadline_round: Optional[int] = None
    log_directory: str

    def __init__(self, **data):
        if 'deadline_time' not in data:
            deadline_time_str = os.getenv("NEGOTIATION_DEADLINE_TIME")
            if deadline_time_str is None:
                raise ValueError("NEGOTIATION_DEADLINE_TIME environment variable is required")
            data['deadline_time'] = int(deadline_time_str)

        if 'log_directory' not in data:
            log_directory = os.getenv("SESSION_LOG_DIR")
            if log_directory is None:
                raise ValueError("SESSION_LOG_DIR environment variable is required")
            data['log_directory'] = log_directory

        super().__init__(**data)

    @field_validator('deadline_time')
    @classmethod
    def validate_deadline(cls, v):
        if v <= 0:
            raise ValueError("Deadline time must be positive")
        return v


class DomainConfig(BaseModel):
    """Domain and profile configuration."""

    domain_name: str
    human_profile: str
    llm_profile: str

    def __init__(self, **data):
        if 'domain_name' not in data:
            domain_name = os.getenv("DOMAIN_NAME")
            if domain_name is None:
                # Default to holiday if not specified (dynamic domains handling)
                domain_name = "holiday"
            data['domain_name'] = domain_name

        if 'human_profile' not in data:
            human_profile = os.getenv("HUMAN_PROFILE")
            if human_profile is None:
                human_profile = "profileA.json"
            data['human_profile'] = human_profile

        if 'llm_profile' not in data:
            llm_profile = os.getenv("LLM_PROFILE")
            if llm_profile is None:
                llm_profile = "profileB.json"
            data['llm_profile'] = llm_profile

        super().__init__(**data)

    @property
    def human_profile_path(self) -> str:
        # Default to englisch for startup; actual language is determined per-session
        language_dir = getattr(self, 'language_dir', 'englisch')
        return f"main/domains/{language_dir}/{self.domain_name}/{self.human_profile}"

    @property
    def llm_profile_path(self) -> str:
        # Default to englisch for startup; actual language is determined per-session
        language_dir = getattr(self, 'language_dir', 'englisch')
        return f"main/domains/{language_dir}/{self.domain_name}/{self.llm_profile}"


class ModelConfig(BaseModel):
    """LLM model configuration. Single shared model name used by every
    LLM component (BiddingStrategy, DialogueGeneration, MessageClassifier).
    Per-component temperature/max_tokens/reasoning live in env vars read
    by `main.llm_components.base_component.Component`.
    """

    model_name: str

    def __init__(self, **data):
        if 'model_name' not in data:
            model_name = os.getenv("LLM_MODEL_NAME")
            if model_name is None or model_name == "":
                raise ValueError("LLM_MODEL_NAME environment variable is required")
            data['model_name'] = model_name
        super().__init__(**data)


class AppConfig(BaseModel):
    """Complete application configuration."""

    session: SessionConfig
    domain: DomainConfig
    model: ModelConfig
    langsmith_project: str

    def __init__(self, **data):
        if 'session' not in data:
            data['session'] = SessionConfig()
        if 'domain' not in data:
            data['domain'] = DomainConfig()
        if 'model' not in data:
            data['model'] = ModelConfig()
        if 'langsmith_project' not in data:
            langsmith_project = os.getenv("LANGSMITH_PROJECT")
            if langsmith_project is None:
                raise ValueError("LANGSMITH_PROJECT environment variable is required")
            data['langsmith_project'] = langsmith_project

        super().__init__(**data)
