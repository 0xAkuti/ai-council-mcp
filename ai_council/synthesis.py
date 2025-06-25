import random
import re
from typing import List, Tuple, Optional
from .models import ModelConfig, ModelManager
from .logger import AICouncilLogger


class ResponseSynthesizer:
    """Handles synthesis of multiple model responses into a final answer."""
    
    def __init__(self, model_manager: ModelManager, logger: Optional[AICouncilLogger] = None):
        self.model_manager = model_manager
        self.logger = logger or AICouncilLogger()
    
    def create_synthesis_prompt(
        self, 
        context: str, 
        question: str, 
        responses: List[str], 
        models: List[ModelConfig]
    ) -> str:
        """Create the synthesis prompt using anonymous code names."""
        if len(responses) != len(models):
            raise ValueError(f"Mismatch between responses ({len(responses)}) and models ({len(models)})")
        
        # Filter out error responses for synthesis
        valid_pairs = [(response, model) for response, model in zip(responses, models) 
                      if not response.startswith("Error from") and not response.startswith("Timeout error")]
        
        if not valid_pairs:
            raise ValueError("No valid responses available for synthesis")
        
        valid_responses, valid_models = zip(*valid_pairs)
        
        prompt = f"""This is a high-level reasoning task. Your goal is to act as a critical and objective evaluator of the provided model outputs. Do not simply repeat the information; your value is in the synthesis and analysis.

**Original Context:**
> {context}

**Original Question:**
> {question}

**Analysis Task:**
You have been provided with responses to the above question from different AI systems ({', '.join([m.code_name for m in valid_models])}). Your task is to critically evaluate these responses and generate a single, comprehensive, and well-reasoned final answer.

IMPORTANT: Use only the provided system code names ({', '.join([m.code_name for m in valid_models])}) when referring to the systems. Do not speculate about their actual identities.

Please structure your response by following these steps:

1. **Identify Areas of Agreement:**
   * Begin by summarizing the key points, conclusions, or facts where all systems are in agreement. This will form the foundation of the final answer.

2. **Identify Areas of Disagreement and Nuance:**
   * Carefully compare the responses and highlight any contradictions, discrepancies, or subtle differences in their conclusions or the data they provided.
   * For each point of disagreement, briefly analyze why the systems might have differed.
   * Evaluate all inputs with equal weight, without bias toward any particular system.

3. **Synthesize a Final, Verified Answer:**
   * Based on your analysis of the agreements and disagreements, construct what you believe to be the most accurate and complete answer.
   * If one system's answer seems more plausible or well-supported, explain why using only the code names.
   * If the systems missed something important from the original context, please add it.
   * Present this final answer clearly and concisely.

**System Responses for Analysis:**
---"""
        
        for response, model in valid_pairs:
            prompt += f"\n**{model.code_name} Response:**\n> {response}\n---"
        
        prompt += "\n\n**Final Synthesized Answer:**\n(Begin your final answer here, following the three steps outlined in the Analysis Task.)"
        
        return prompt.strip()
    
    def replace_code_names_with_real_names(self, text: str, models: List[ModelConfig]) -> str:
        """Replace anonymous code names with real model names in the final synthesis."""
        result = text
        for model in models:
            # Use word boundaries to avoid partial replacements
            code_name_pattern = r'\b' + re.escape(model.code_name) + r'\b'
            result = re.sub(code_name_pattern, model.name, result)
        return result
    
    def select_synthesizer_model(self, models: List[ModelConfig]) -> ModelConfig:
        """Select which model will act as the synthesizer."""
        if not models:
            raise ValueError("No models available for synthesis")
        
        selection_method = self.model_manager.config.synthesis_model_selection
        
        if selection_method == "random":
            return random.choice(models)
        elif selection_method == "first":
            return models[0]
        else:  # Default to random
            return random.choice(models)
    
    async def synthesize_responses(
        self,
        context: str,
        question: str,
        responses: List[str],
        models: List[ModelConfig]
    ) -> Tuple[str, ModelConfig]:
        """Synthesize multiple model responses into a final answer.
        
        Returns:
            Tuple of (final_synthesis, selected_synthesizer_model)
        """
        if not responses or not models:
            raise ValueError("No responses or models provided for synthesis")
        
        if len(responses) != len(models):
            raise ValueError(f"Mismatch between responses ({len(responses)}) and models ({len(models)})")
        
        # Select synthesizer model
        synthesizer_model = self.select_synthesizer_model(models)
        self.logger.info(f"Selected {synthesizer_model.name} as synthesizer", {
            "synthesizer": synthesizer_model.name,
            "synthesizer_code_name": synthesizer_model.code_name
        })
        
        # Create synthesis prompt
        try:
            synthesis_prompt = self.create_synthesis_prompt(context, question, responses, models)
        except ValueError as e:
            self.logger.error(f"Failed to create synthesis prompt: {e}")
            # Fallback: return the best available response
            valid_responses = [r for r in responses if not r.startswith("Error from") and not r.startswith("Timeout error")]
            if valid_responses:
                return valid_responses[0], synthesizer_model
            else:
                return "All models failed to provide valid responses.", synthesizer_model
        
        self.logger.info("Generating final synthesis...", {
            "prompt_length": len(synthesis_prompt),
            "synthesizer_model": synthesizer_model.name
        })
        
        # Get synthesis response
        raw_synthesis = await self.model_manager.call_model(
            synthesizer_model, 
            "",  # Empty context for synthesis
            synthesis_prompt,  # Full synthesis prompt
            is_synthesis=True
        )
        
        # Check if synthesis failed
        if raw_synthesis.startswith("Error from") or raw_synthesis.startswith("Timeout error"):
            self.logger.error(f"Synthesis failed: {raw_synthesis}")
            # Fallback: return the first valid response
            valid_responses = [r for r in responses if not r.startswith("Error from") and not r.startswith("Timeout error")]
            if valid_responses:
                return f"Synthesis failed, using fallback response:\n\n{valid_responses[0]}", synthesizer_model
            else:
                return "All models failed to provide valid responses, including synthesis.", synthesizer_model
        
        # Replace code names with real model names
        final_synthesis = self.replace_code_names_with_real_names(raw_synthesis, models)
        
        self.logger.info("Final synthesis completed and de-anonymized", {
            "synthesis_length": len(final_synthesis)
        })
        
        return final_synthesis, synthesizer_model 