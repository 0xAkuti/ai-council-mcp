"""
Pydantic-based configuration system for AI Council.

Follows Pydantic v2 best practices with proper BaseSettings usage.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


class SynthesisModelSelection(str, Enum):
    """Synthesis model selection strategy."""
    RANDOM = "random"
    FIRST = "first"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelConfig(BaseModel):
    """Configuration for a single AI model."""
    name: str = Field(..., min_length=1, description="Human-readable name of the model")
    model_id: str = Field(..., min_length=1, description="Provider-specific model identifier")
    provider: Provider = Field(default=Provider.OPENROUTER, description="AI provider")
    base_url: Optional[str] = Field(default=None, description="Custom OpenAI-compatible API base URL (overrides provider default)")
    api_key: Optional[str] = Field(default=None, description="API key for this specific model (overrides global keys)")
    code_name: Optional[str] = Field(default=None, description="Anonymous code name for bias reduction (auto-assigned if not provided)")
    enabled: bool = Field(default=True, description="Whether this model is enabled")


# Default code names for anonymous model identification
DEFAULT_CODE_NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", 
    "Zeta", "Eta", "Theta", "Iota", "Kappa"
]


class AICouncilConfig(BaseSettings):
    """Main configuration class for AI Council using BaseSettings for environment support."""
    
    model_config = SettingsConfigDict(
        env_prefix="AI_COUNCIL_",
        case_sensitive=False,
        extra="ignore"
    )

    # API Keys, set alias to disable env prefix
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key", alias="openai_api_key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key", alias="openrouter_api_key")

    # Settings with validation
    max_models: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of models to consult simultaneously"
    )
    parallel_timeout: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Timeout for parallel API calls in seconds"
    )
    synthesis_model_selection: SynthesisModelSelection = Field(
        default=SynthesisModelSelection.RANDOM,
        description="Strategy for selecting synthesis model"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )

    # Models configuration
    models: List[ModelConfig] = Field(
        default_factory=list,
        description="List of AI models to use"
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and setup."""
        # Set default models if none provided
        if not self.models:
            self.models = self._get_default_models()
        
        # Validate model count limit
        if len(self.models) > 10:
            raise ValueError(f"Cannot configure more than 10 models (found {len(self.models)})")
        
        # Auto-assign code names if not provided
        self._assign_code_names()
        
        # Validate unique code names
        code_names = [model.code_name for model in self.models if model.code_name]
        if len(code_names) != len(set(code_names)):
            raise ValueError("Duplicate code names found in model configuration")
        
        # Ensure at least two enabled models
        enabled_models = [model for model in self.models if model.enabled]
        if len(enabled_models) < 2:
            raise ValueError("At least two models must be enabled")
        
        self._validate_api_key_requirements(enabled_models)

    def _assign_code_names(self) -> None:
        """Auto-assign code names to models that don't have them."""
        available_names = DEFAULT_CODE_NAMES.copy()
        
        # First pass: remove already used code names from available list
        for model in self.models:
            if model.code_name and model.code_name in available_names:
                available_names.remove(model.code_name)
        
        # Second pass: assign code names to models without them
        for name_index, model in enumerate(self.models):
            if not model.code_name:
                model.code_name = available_names[name_index]

    def _get_default_models(self) -> List[ModelConfig]:
        """Get default model configuration for uvx usage."""
        return [
            ModelConfig(
                name="claude-sonnet-4",
                provider=Provider.OPENROUTER,
                model_id="anthropic/claude-sonnet-4",
                enabled=True
            ),
            ModelConfig(
                name="gemini-2.5-pro",
                provider=Provider.OPENROUTER,
                model_id="google/gemini-2.5-pro",
                enabled=True
            ),
            ModelConfig(
                name="deepseek-chat-v3",
                provider=Provider.OPENROUTER,
                model_id="deepseek/deepseek-chat-v3-0324",
                enabled=True
            )
        ]

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models up to max_models limit."""
        enabled = [model for model in self.models if model.enabled]
        return enabled[:self.max_models]

    def get_log_level(self) -> int:
        """Get logging level as integer constant."""
        return getattr(logging, self.log_level.value)

    def _validate_api_key_requirements(self, enabled_models: List["ModelConfig"]) -> None:
        """Validate that required API keys are available for enabled models."""
        
        for model in enabled_models:
            if model.api_key:
                continue
            elif model.provider == Provider.CUSTOM:
                if not model.base_url:
                    raise ValueError("Custom endpoints require a base_url")
                if not model.api_key:
                    raise ValueError("Custom endpoints require an api_key")
            elif model.provider == Provider.OPENAI and not self.openai_api_key:
                raise ValueError("OpenAI API key is required if using OpenAI models")
            elif model.provider == Provider.OPENROUTER and not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is required if using OpenRouter models")

def load_config(
    config_file: Optional[str] = None,
    **overrides
) -> AICouncilConfig:
    """
    Load configuration from file and environment with overrides.
    
    Args:
        config_file: Optional path to YAML config file
        **overrides: Direct field overrides
    
    Returns:
        AICouncilConfig instance
    """
    # Find config file if not specified
    if config_file is None:
        default_path = Path.home() / ".config" / "ai-council" / "config.yaml"
        if default_path.exists():
            config_file = str(default_path)
    
    # Load from YAML file if it exists
    yaml_data = {}
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load config file {config_file}: {e}")
    
    # Merge YAML data with overrides (overrides take precedence)
    config_data = {**yaml_data, **overrides}

    return AICouncilConfig(**config_data)