import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class AICouncilLogger:
    """Logger for AI Council with file and console output."""
    
    def __init__(self):
        timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
        log_dir = os.environ.get("TMPDIR") or tempfile.gettempdir()
        self.log_file = Path(log_dir) / f"ai-council-{timestamp}.log"
        self.log(f"=== AI Council Session Started, logging to {self.log_file} ===")
    
    def log(self, message: str, data: Optional[Any] = None) -> None:
        """Log a message with optional data to both file and console."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}"
        
        if data is not None:
            log_entry += f"\n{json.dumps(data, indent=2, default=str)}"
        
        log_entry += "\n"
        
        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        # Also log to console (stderr to not interfere with MCP protocol)
        import sys
        print(f"[LOG] {message}", file=sys.stderr, flush=True)
    
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