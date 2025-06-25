import json
import logging
from datetime import datetime
from typing import Any, Optional


class AICouncilLogger:
    """Singleton logger for AI Council using standard Python logging."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            # Set up the logger
            self.logger = logging.getLogger('ai_council')
            self.logger.setLevel(logging.INFO)
            
            # Create console handler if not already present
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            
            self.info("AI Council Session Started")
    
    def log(self, message: str, data: Optional[Any] = None) -> None:
        """Log a message with optional data using INFO level."""
        self.info(message, data)
    
    def debug(self, message: str, data: Optional[Any] = None) -> None:
        """Log a debug message with optional data."""
        if data is not None:
            data_str = json.dumps(data, indent=2, default=str)
            self.logger.debug(f"{message}\n{data_str}")
        else:
            self.logger.debug(message)
    
    def info(self, message: str, data: Optional[Any] = None) -> None:
        """Log an info message with optional data."""
        if data is not None:
            data_str = json.dumps(data, indent=2, default=str)
            self.logger.info(f"{message}\n{data_str}")
        else:
            self.logger.info(message)
    
    def warning(self, message: str, data: Optional[Any] = None) -> None:
        """Log a warning message with optional data."""
        if data is not None:
            data_str = json.dumps(data, indent=2, default=str)
            self.logger.warning(f"{message}\n{data_str}")
        else:
            self.logger.warning(message)
    
    def error(self, message: str, data: Optional[Any] = None) -> None:
        """Log an error message with optional data."""
        if data is not None:
            data_str = json.dumps(data, indent=2, default=str)
            self.logger.error(f"{message}\n{data_str}")
        else:
            self.logger.error(message)
    
    def set_level(self, level: int) -> None:
        """Set the logging level using logging constants (e.g., logging.DEBUG, logging.INFO)."""
        self.logger.setLevel(level) 