#!/usr/bin/env python3

import asyncio
import json
import time
import sys
from typing import Any, Dict, List, Optional, Union

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.types as types
from pydantic import AnyUrl, BaseModel

from .models import ModelManager, ConfigValidationError
from .synthesis import ResponseSynthesizer
from .logger import AICouncilLogger
from .config import load_config


# Response Models
class ConsensusInfo(BaseModel):
    """Information about model consensus."""
    models_queried: int
    models_succeeded: int
    models_failed: int


class SuccessData(BaseModel):
    """Data returned on successful AI Council processing."""
    answer: str
    consensus: ConsensusInfo


class ErrorInfo(BaseModel):
    """Error information structure."""
    code: str
    message: str
    type: str
    details: str


class SuccessResponse(BaseModel):
    """Successful response from AI Council."""
    status: str = "success"
    data: SuccessData


class ErrorResponse(BaseModel):
    """Error response from AI Council."""
    status: str = "error"
    error: ErrorInfo
    data: Optional[Dict[str, Any]] = None


# Union type for all possible responses
AICouncilResponse = Union[SuccessResponse, ErrorResponse]


class AICouncilServer:
    """Main MCP server for AI Council."""
    
    def __init__(self, config=None):
        self.logger = AICouncilLogger()
        try:
            # Load config if not provided
            if config is None:
                config = load_config()
            
            self.config = config
            self.model_manager = ModelManager(config=config, logger=self.logger)
            self.synthesizer = ResponseSynthesizer(self.model_manager, logger=self.logger)
        except (ConfigValidationError, ValueError) as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Council Server: {e}")
            raise
        
        self.server = Server("ai-council")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="ai_council",
                    description="A tool that consults multiple AI models in parallel, then uses one of them to synthesize the results into a single, high-quality answer. Use this for complex questions requiring deep analysis and verification.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Important background information and context for the problem to be solved."
                            },
                            "question": {
                                "type": "string", 
                                "description": "The specific, detailed question you want to be answered."
                            }
                        },
                        "required": ["context", "question"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            if name != "ai_council":
                error_result = ErrorResponse(
                    error=ErrorInfo(
                        code="UNKNOWN_TOOL",
                        message=f"Unknown tool: {name}",
                        type="user_input_error",
                        details=f"The tool '{name}' is not supported. Available tools: ai_council"
                    )
                )
                return [types.TextContent(type="text", text=error_result.model_dump_json(indent=2))]
            
            try:
                result = await self._process_ai_council(arguments)
                return [types.TextContent(type="text", text=result.model_dump_json(indent=2))]
            except Exception as e:
                self.logger.error(f"Error in tool call: {e}")
                error_result = ErrorResponse(
                    error=ErrorInfo(
                        code="INTERNAL_ERROR",
                        message="An unexpected error occurred during processing",
                        type="system_error",
                        details=str(e)
                    )
                )
                return [types.TextContent(type="text", text=error_result.model_dump_json(indent=2))]
    
    def _validate_input(self, context: str, question: str) -> None:
        """Validate input parameters."""
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Basic length validation
        if len(context) > 10000:
            raise ValueError("Context too long (max 10,000 characters)")
        
        if len(question) > 5000:
            raise ValueError("Question too long (max 5,000 characters)")
    
    async def _process_ai_council(self, arguments: Dict[str, Any]) -> AICouncilResponse:
        """Process the AI council request."""
        start_time = time.time()
        
        # Validate arguments
        context = arguments.get("context", "")
        question = arguments.get("question", "")
        
        # Validate input parameters
        try:
            self._validate_input(context, question)
        except ValueError as e:
            return ErrorResponse(
                error=ErrorInfo(
                    code="INVALID_INPUT",
                    message="Input validation failed",
                    type="user_input_error",
                    details=str(e)
                )
            )
        
        # Get enabled models
        models = self.model_manager.get_enabled_models()
        if not models:
            return ErrorResponse(
                error=ErrorInfo(
                    code="NOT_ENOUGH_MODELS_ENABLED",
                    message="Not enough models enabled in configuration",
                    type="configuration_error",
                    details="At least two models must be enabled in the configuration to process requests"
                )
            )
        
        self.logger.info("Starting AI Council process...", {
            "models": [{"name": m.name, "code_name": m.code_name} for m in models],
            "model_count": len(models)
        })
        
        # Make parallel calls to all models
        self.logger.info("Dispatching calls to all models in parallel")
        parallel_start = time.time()
        
        responses = await self.model_manager.call_models_parallel(models, context, question)
        parallel_duration = time.time() - parallel_start
        
        self.logger.info(f"All model responses received in {parallel_duration:.2f}s", {
            "parallel_duration": parallel_duration,
            "response_lengths": [len(r) for r in responses]
        })
        
        # Check if we have any valid responses
        valid_responses = [r for r in responses if not r.startswith("Error from") and not r.startswith("Timeout error")]
        if not valid_responses:
            return ErrorResponse(
                error=ErrorInfo(
                    code="ALL_MODELS_FAILED",
                    message="All models failed to provide valid responses",
                    type="service_error",
                    details=f"Attempted to call {len(models)} models but all failed or timed out"
                ),
                data={
                    "attempted_models": len(models),
                    "failed_responses": len(responses)
                }
            )
        
        if len(valid_responses) < len(responses):
            self.logger.warning(f"Only {len(valid_responses)} out of {len(responses)} models provided valid responses")
        
        # Synthesize responses
        synthesis_start = time.time()
        final_synthesis, selected_synthesizer = await self.synthesizer.synthesize_responses(
            context, question, responses, models
        )
        synthesis_duration = time.time() - synthesis_start
        self.logger.info(f"Synthesis completed in {synthesis_duration:.2f}s", {
            "synthesis_duration": synthesis_duration,
            "synthesizer_model": selected_synthesizer.name
        })
        
        # Prepare result
        total_duration = time.time() - start_time
        result = SuccessResponse(
            data=SuccessData(
                answer=final_synthesis,
                consensus=ConsensusInfo(
                    models_queried=len(models),
                    models_succeeded=len(valid_responses),
                    models_failed=len(responses) - len(valid_responses)
                )
            )
        )
        
        self.logger.info("Process completed successfully", {
            "total_duration": total_duration
        })
        
        return result
    
    async def run(self):
        """Run the MCP server."""
        # MCP server setup
        from mcp.server.stdio import stdio_server
        
        self.logger.info("Starting AI Council MCP Server on stdio")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ai-council",
                    server_version="0.2.2",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


def main():
    """Main entry point."""
    import argparse
    
    # Simple argument parsing for basic options
    parser = argparse.ArgumentParser(description="AI Council MCP Server")
    # add api keys
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    parser.add_argument("--openrouter-api-key", help="OpenRouter API key")
    parser.add_argument("--max-models", type=int, help="Maximum number of models to use")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--parallel-timeout", type=int, help="Timeout for parallel calls")
    
    args = parser.parse_args()
    
    async def async_main():
        try:
            # Create config with command line overrides
            config_kwargs = {}
            if args.openai_api_key:
                config_kwargs['openai_api_key'] = args.openai_api_key
            if args.openrouter_api_key:
                config_kwargs['openrouter_api_key'] = args.openrouter_api_key
            if args.max_models:
                config_kwargs['max_models'] = args.max_models
            if args.log_level:
                config_kwargs['log_level'] = args.log_level
            if args.parallel_timeout:
                config_kwargs['parallel_timeout'] = args.parallel_timeout
            
            config = load_config(config_file=args.config, **config_kwargs)
            
            server = AICouncilServer(config=config)
            await server.run()
        except (ConfigValidationError, ValueError) as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Failed to start AI Council server: {e}", file=sys.stderr)
            sys.exit(1)
    
    asyncio.run(async_main())


if __name__ == "__main__":
    main() 