from starlette.config import Config
from pathlib import Path

# HS Base directory path
# AA Set up the base directory and path for the environment file
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = Path.joinpath(BASE_DIR, ".env")
print(f'ENV_PATH {type(ENV_PATH)}: {ENV_PATH}')
config = Config(ENV_PATH)

# Load environment configuration
OPENAI_API_KEY = config("OPENAI_API_KEY", cast=str, default='DEV')
print(f'OPENAI_API_KEY {type(OPENAI_API_KEY)}: {OPENAI_API_KEY}')
GEMINI_API_KEY = config("GEMINI_API_KEY", cast=str, default='DEV')