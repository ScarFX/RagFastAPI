import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time, os


def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    env = os.getenv("DEBUG_MODE","false")
    if env.lower() == "true":
        logger.setLevel(level)
    else:
        logger.setLevel(0)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
def get_env_variable(var_name: str, default_value: str = None, required: bool = False) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None and required:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value

class LogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        self.logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")
        return response