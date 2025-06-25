import asyncio
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import yaml
from openai import OpenAI, AsyncOpenAI
import httpx
from .logger import AICouncilLogger


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str
    model_id: str
    code_name: str
    enabled: bool


class ModelManager:
    """Manages model configurations and API calls."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = AICouncilLogger()
        self.config = self._load_config(config_path)
        self.openai_client = None
        self.openrouter_client = None
        self._init_clients()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Expand environment variables
            for key, value in config.get("api_keys", {}).items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config["api_keys"][key] = os.environ.get(env_var, "")
            
            return config
        except Exception as e:
            self.logger.log(f"Error loading config: {e}")
            return {"models": [], "settings": {}, "api_keys": {}}
    
    def _init_clients(self) -> None:
        """Initialize API clients."""
        api_keys = self.config.get("api_keys", {})
        
        # OpenAI client
        openai_key = api_keys.get("openai_api_key", "")
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
        else:
            self.logger.log("Warning: No OpenAI API key found")
        
        # OpenRouter client (using OpenAI format)
        openrouter_key = api_keys.get("openrouter_api_key", "")
        if openrouter_key:
            self.openrouter_client = AsyncOpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            self.logger.log("Warning: No OpenRouter API key found")
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models up to max_models limit."""
        models = []
        max_models = self.config.get("settings", {}).get("max_models", 3)
        
        for model_data in self.config.get("models", []):
            if model_data.get("enabled", False) and len(models) < max_models:
                models.append(ModelConfig(**model_data))
        
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
            
            # Make the API call
            response = await client.chat.completions.create(
                model=model_config.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content or ""
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
        timeout = self.config.get("settings", {}).get("parallel_timeout", 120)
        
        try:
            tasks = [
                self.call_model(model, context, question) 
                for model in models
            ]
            
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=timeout
            )
            
            return responses
            
        except asyncio.TimeoutError:
            self.logger.log(f"Parallel calls timed out after {timeout}s")
            return [f"Timeout error for model {model.name}" for model in models]
        except Exception as e:
            self.logger.log(f"Error in parallel calls: {e}")
            return [f"Error for model {model.name}: {str(e)}" for model in models] 