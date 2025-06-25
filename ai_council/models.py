import asyncio
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import yaml
from openai import AsyncOpenAI
from .logger import AICouncilLogger


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str
    model_id: str
    code_name: str
    enabled: bool


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ModelManager:
    """Manages model configurations and API calls."""
    
    def __init__(self, config_path: str = "config.yaml", logger: Optional[AICouncilLogger] = None):
        self.logger = logger or AICouncilLogger()
        self.config = self._load_config(config_path)
        self._validate_config()
        self.openai_client = None
        self.openrouter_client = None
        self._init_clients()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Expand environment variables
            api_keys = config.get("api_keys", {})
            for key, value in api_keys.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    api_keys[key] = os.environ.get(env_var, "")
            
            return config
        except FileNotFoundError:
            self.logger.log(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except yaml.YAMLError as e:
            self.logger.log(f"Error parsing YAML config: {e}")
            raise ConfigValidationError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            self.logger.log(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config file is missing or invalid."""
        return {
            "models": [],
            "settings": {
                "max_models": 3,
                "parallel_timeout": 120,
                "synthesis_model_selection": "random",
                "log_level": "INFO"
            },
            "api_keys": {
                "openai_api_key": "",
                "openrouter_api_key": ""
            }
        }
    
    def _validate_config(self) -> None:
        """Validate configuration structure and values."""
        required_sections = ["models", "settings", "api_keys"]
        for section in required_sections:
            if section not in self.config:
                raise ConfigValidationError(f"Missing required config section: {section}")
        
        # Validate models
        models = self.config.get("models", [])
        if not isinstance(models, list):
            raise ConfigValidationError("Models must be a list")
        
        code_names = set()
        for i, model in enumerate(models):
            required_fields = ["name", "provider", "model_id", "code_name", "enabled"]
            for field in required_fields:
                if field not in model:
                    raise ConfigValidationError(f"Model {i} missing required field: {field}")
            
            # Check for duplicate code names
            code_name = model["code_name"]
            if code_name in code_names:
                raise ConfigValidationError(f"Duplicate code name: {code_name}")
            code_names.add(code_name)
            
            # Validate provider
            if model["provider"] not in ["openai", "openrouter"]:
                raise ConfigValidationError(f"Unknown provider: {model['provider']}")
        
        # Validate settings
        settings = self.config.get("settings", {})
        max_models = settings.get("max_models", 3)
        if not isinstance(max_models, int) or max_models < 1:
            raise ConfigValidationError("max_models must be a positive integer")
        
        timeout = settings.get("parallel_timeout", 120)
        if not isinstance(timeout, int) or timeout < 1:
            raise ConfigValidationError("parallel_timeout must be a positive integer")
    
    def _init_clients(self) -> None:
        """Initialize API clients."""
        api_keys = self.config.get("api_keys", {})
        
        # OpenAI client
        openai_key = api_keys.get("openai_api_key", "")
        if openai_key and openai_key != "empty":
            try:
                self.openai_client = AsyncOpenAI(api_key=openai_key)
            except Exception as e:
                self.logger.log(f"Failed to initialize OpenAI client: {e}")
        else:
            self.logger.log("Warning: No OpenAI API key found")
        
        # OpenRouter client (using OpenAI format)
        openrouter_key = api_keys.get("openrouter_api_key", "")
        if openrouter_key and openrouter_key != "empty":
            try:
                self.openrouter_client = AsyncOpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            except Exception as e:
                self.logger.log(f"Failed to initialize OpenRouter client: {e}")
        else:
            self.logger.log("Warning: No OpenRouter API key found")
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models up to max_models limit."""
        models = []
        max_models = self.config.get("settings", {}).get("max_models", 3)
        
        for model_data in self.config.get("models", []):
            if model_data.get("enabled", False) and len(models) < max_models:
                try:
                    models.append(ModelConfig(**model_data))
                except Exception as e:
                    self.logger.log(f"Error creating model config: {e}")
        
        return models
    
    async def call_model(
        self, 
        model_config: ModelConfig, 
        context: str, 
        question: str,
        is_synthesis: bool = False
    ) -> str:
        """Make an API call to a specific model."""
        start_time = time.time()
        self.logger.log(f"Calling {model_config.name}...", {
            "model": model_config.name,
            "code_name": model_config.code_name
        })
        
        try:
            # Input validation
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")
            
            # For synthesis calls, use the question as the full prompt
            if is_synthesis:
                prompt = question
            else:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease provide a detailed, well-reasoned answer."
            
            # Choose the appropriate client
            if model_config.provider == "openai":
                client = self.openai_client
            elif model_config.provider == "openrouter":
                client = self.openrouter_client
            else:
                raise ValueError(f"Unknown provider: {model_config.provider}")
            
            if not client:
                raise ValueError(f"No client available for provider: {model_config.provider}")
            
            # Make the API call with better error handling
            try:
                response = await client.chat.completions.create(
                    model=model_config.model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("Empty response received from model")
                
            except Exception as api_error:
                # More specific API error handling
                error_msg = str(api_error)
                if "rate_limit" in error_msg.lower():
                    raise ValueError(f"Rate limit exceeded for {model_config.name}")
                elif "auth" in error_msg.lower():
                    raise ValueError(f"Authentication failed for {model_config.name}")
                else:
                    raise ValueError(f"API error for {model_config.name}: {error_msg}")
            
            duration = time.time() - start_time
            
            self.logger.log(f"Received response from {model_config.name} in {duration:.2f}s", {
                "model": model_config.name,
                "duration": duration,
                "response_length": len(content),
                "response_preview": content[:200] + "..." if len(content) > 200 else content
            })
            
            return content
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error from {model_config.name}: {str(e)}"
            self.logger.log(f"Error calling {model_config.name}", {
                "model": model_config.name,
                "error": str(e),
                "duration": duration
            })
            return error_msg
    
    async def call_models_parallel(
        self, 
        models: List[ModelConfig], 
        context: str, 
        question: str
    ) -> List[str]:
        """Call multiple models in parallel."""
        if not models:
            raise ValueError("No models provided for parallel calls")
        
        timeout = self.config.get("settings", {}).get("parallel_timeout", 120)
        
        try:
            tasks = [
                self.call_model(model, context, question) 
                for model in models
            ]
            
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Handle exceptions in responses
            final_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    error_msg = f"Error for model {models[i].name}: {str(response)}"
                    final_responses.append(error_msg)
                else:
                    final_responses.append(response)
            
            return final_responses
            
        except asyncio.TimeoutError:
            self.logger.log(f"Parallel calls timed out after {timeout}s")
            return [f"Timeout error for model {model.name}" for model in models]
        except Exception as e:
            self.logger.log(f"Error in parallel calls: {e}")
            return [f"Error for model {model.name}: {str(e)}" for model in models] 