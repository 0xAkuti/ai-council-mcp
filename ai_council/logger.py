import json
import sys
from datetime import datetime
from typing import Any, Optional


class AICouncilLogger:
    """Singleton logger for AI Council with console output."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.log("=== AI Council Session Started ===")
    
    def log(self, message: str, data: Optional[Any] = None) -> None:
        """Log a message with optional data to console."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] {message}"
            
            # Log to console (stderr to not interfere with MCP protocol)
            print(f"[LOG] {message}", file=sys.stderr, flush=True)
            
            if data is not None:
                data_str = json.dumps(data, indent=2, default=str)
                print(f"[LOG-DATA]\n{data_str}", file=sys.stderr, flush=True)
            
        except Exception as e:
            # Emergency fallback logging
            print(f"[LOG-CRITICAL] Logger error: {e}", file=sys.stderr, flush=True) 