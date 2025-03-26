# utils/logger.py
import logging
import sys
import time
from datetime import datetime
import config

# Configure logging
logger = logging.getLogger("llm-medusa-service")
logger.setLevel(getattr(logging, config.LOG_LEVEL))

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Performance logging
class Timer:
    """Utility class for timing code execution."""
    
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.name:
            logger.info(f"{self.name} completed in {elapsed:.4f}s")
        return False
    
    @property
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time