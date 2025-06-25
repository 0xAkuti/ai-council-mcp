#!/usr/bin/env python3

import asyncio
import json
import time
import sys
from typing import Any, Dict, List

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
from pydantic import AnyUrl

from .models import ModelManager, ConfigValidationError
from .synthesis import ResponseSynthesizer
from .logger import AICouncilLogger


class AICouncilServer:
    """Main MCP server for AI Council."""
    
    def __init__(self):
        self.logger = AICouncilLogger()
        try:
            self.model_manager = ModelManager(logger=self.logger)
            self.synthesizer = ResponseSynthesizer(self.model_manager, logger=self.logger)
        except ConfigValidationError as e:
            self.logger.log(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            self.logger.log(f"Failed to initialize AI Council Server: {e}")
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
                raise ValueError(f"Unknown tool: {name}")
            
            try:
                result = await self._process_ai_council(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                self.logger.log(f"Error in tool call: {e}")
                error_result = {
                    "error": str(e),
                    "debug": {
                        "log_file": self.logger.get_log_path()
                    },
                    "status": "failed"
                }
                return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
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
    
    async def _process_ai_council(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Process the AI council request."""
        start_time = time.time()
        
        # Validate arguments
        context = arguments.get("context", "")
        question = arguments.get("question", "")
        
        self._validate_input(context, question)
        
        # Get enabled models
        models = self.model_manager.get_enabled_models()
        if not models:
            raise ValueError("No enabled models found in configuration")
        
        self.logger.log("Starting AI Council process...", {
            "models": [{"name": m.name, "code_name": m.code_name} for m in models],
            "model_count": len(models)
        })
        
        # Make parallel calls to all models
        self.logger.log("Dispatching calls to all models in parallel")
        parallel_start = time.time()
        
        responses = await self.model_manager.call_models_parallel(models, context, question)
        parallel_duration = time.time() - parallel_start
        
        self.logger.log(f"All model responses received in {parallel_duration:.2f}s", {
            "parallel_duration": parallel_duration,
            "response_lengths": [len(r) for r in responses]
        })
        
        # Check if we have any valid responses
        valid_responses = [r for r in responses if not r.startswith("Error from") and not r.startswith("Timeout error")]
        if not valid_responses:
            raise ValueError("All models failed to provide valid responses")
        
        if len(valid_responses) < len(responses):
            self.logger.log(f"Warning: Only {len(valid_responses)} out of {len(responses)} models provided valid responses")
        
        # Synthesize responses
        synthesis_start = time.time()
        final_synthesis = await self.synthesizer.synthesize_responses(
            context, question, responses, models
        )
        synthesis_duration = time.time() - synthesis_start
        
        # Prepare result
        total_duration = time.time() - start_time
        result = {
            "models_used": [m.model_id for m in models],
            "synthesizer_model": self.synthesizer.select_synthesizer_model(models).name,
            "final_synthesis": final_synthesis,
            "timing": {
                "total_duration_ms": int(total_duration * 1000),
                "parallel_duration_ms": int(parallel_duration * 1000),
                "synthesis_duration_ms": int(synthesis_duration * 1000)
            },
            "debug_log_file": self.logger.get_log_path(),
            "status": "success",
            "response_summary": {
                "total_responses": len(responses),
                "valid_responses": len(valid_responses),
                "failed_responses": len(responses) - len(valid_responses)
            }
        }
        
        self.logger.log("Process completed successfully", {
            "total_duration": total_duration,
            "log_path": self.logger.get_log_path()
        })
        
        return result
    
    async def run(self):
        """Run the MCP server."""
        # MCP server setup
        from mcp.server.stdio import stdio_server
        
        self.logger.log("Starting AI Council MCP Server on stdio")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ai-council",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


def main():
    """Main entry point."""
    async def async_main():
        try:
            server = AICouncilServer()
            await server.run()
        except ConfigValidationError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Failed to start AI Council server: {e}", file=sys.stderr)
            sys.exit(1)
    
    asyncio.run(async_main())


if __name__ == "__main__":
    main() 