import asyncio
import time
from typing import List, Optional
from openai import AsyncOpenAI
from .logger import AICouncilLogger
from .config import AICouncilConfig, ModelConfig, Provider, load_config


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ModelManager:
    """Manages model configurations and API calls."""
    
    def __init__(self, config: Optional[AICouncilConfig] = None, logger: Optional[AICouncilLogger] = None):
        self.logger = logger or AICouncilLogger()
        self.config = config or load_config()
        self._apply_log_level()
        self._validate_api_keys()
    
    def _apply_log_level(self) -> None:
        """Apply log level from configuration to the logger."""
        try:
            log_level = self.config.get_log_level()
            self.logger.set_level(log_level)
        except Exception as e:
            self.logger.warning(f"Failed to apply log level: {e}")
            import logging
            self.logger.set_level(logging.INFO)
    
    def _validate_api_keys(self) -> None:
        """Validate that required API keys are available."""
        if self.config.openai_api_key:
            self.logger.debug("OpenAI API key found")
        else:
            self.logger.warning("No OpenAI API key found")
        
        if self.config.openrouter_api_key:
            self.logger.debug("OpenRouter API key found")
        else:
            self.logger.warning("No OpenRouter API key found")
    
    def _get_client_for_model(self, model_config: ModelConfig) -> AsyncOpenAI:
        """Create an appropriate client for the given model configuration."""
        # Determine which API key to use (priority: model-specific -> provider-specific)
        api_key = ""
        
        if model_config.api_key:
            # prefer to use model-specific API key if provided
            api_key = model_config.api_key
        elif model_config.provider == Provider.CUSTOM:
            api_key = model_config.api_key
            if not api_key:
                raise ValueError(f"API key required for model {model_config.name} using custom endpoint.") 
        elif model_config.provider == Provider.OPENAI:
            api_key = self.config.openai_api_key
            if not api_key:
                raise ValueError(f"OpenAI API key required for model {model_config.name}")
        elif model_config.provider == Provider.OPENROUTER:
            api_key = self.config.openrouter_api_key
            if not api_key:
                raise ValueError(f"OpenRouter API key required for model {model_config.name}")
        else:
            raise ValueError(f"Unknown provider: {model_config.provider}")
        
        # Determine base URL
        if model_config.provider == Provider.CUSTOM:
            base_url = model_config.base_url
        elif model_config.provider == Provider.OPENROUTER:
            base_url = "https://openrouter.ai/api/v1"
        else:
            # OpenAI uses default base URL (None)
            base_url = None
        

        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models up to max_models limit."""
        return self.config.get_enabled_models()
    
    async def call_model(
        self, 
        model_config: ModelConfig, 
        context: str, 
        question: str,
        is_synthesis: bool = False
    ) -> str:
        """Make an API call to a specific model."""
        start_time = time.time()
        self.logger.debug(f"Calling {model_config.name}...", {
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
            client = self._get_client_for_model(model_config)
            
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
            
            self.logger.debug(f"Received response from {model_config.name} in {duration:.2f}s", {
                "model": model_config.name,
                "duration": duration,
                "response_length": len(content),
                "response_preview": content[:200] + "..." if len(content) > 200 else content
            })
            
            return content
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error from {model_config.name}: {str(e)}"
            self.logger.error(f"Error calling {model_config.name}", {
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
        
        timeout = self.config.parallel_timeout
        
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
            self.logger.error(f"Parallel calls timed out after {timeout}s")
            return [f"Timeout error for model {model.name}" for model in models]
        except Exception as e:
            self.logger.error(f"Error in parallel calls: {e}")
            return [f"Error for model {model.name}: {str(e)}" for model in models] 