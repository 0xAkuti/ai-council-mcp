import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class AICouncilLogger:
    """Singleton logger for AI Council with file and console output."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
            log_dir = os.environ.get("TMPDIR") or tempfile.gettempdir()
            self.log_file = Path(log_dir) / f"ai-council-{timestamp}.log"
            self._initialized = True
            self.log(f"=== AI Council Session Started, logging to {self.log_file} ===")
    
    def log(self, message: str, data: Optional[Any] = None) -> None:
        """Log a message with optional data to both file and console."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] {message}"
            
            if data is not None:
                log_entry += f"\n{json.dumps(data, indent=2, default=str)}"
            
            log_entry += "\n"
            
            # Write to file with error handling
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except (IOError, OSError) as e:
                # Fallback to stderr if file writing fails
                import sys
                print(f"[LOG-ERROR] Failed to write to log file: {e}", file=sys.stderr)
            
            # Also log to console (stderr to not interfere with MCP protocol)
            import sys
            print(f"[LOG] {message}", file=sys.stderr, flush=True)
            
        except Exception as e:
            # Emergency fallback logging
            import sys
            print(f"[LOG-CRITICAL] Logger error: {e}", file=sys.stderr, flush=True)
    
    def get_log_path(self) -> str:
        """Get the path to the log file."""
        return str(self.log_file)
    
    def get_log_content(self) -> str:
        """Get the content of the log file."""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except (IOError, OSError) as e:
            self.log(f"Error reading log file: {e}")
            return f"Error reading log file: {e}" 