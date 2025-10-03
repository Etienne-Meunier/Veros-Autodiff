from pathlib import Path
from dotenv import load_dotenv

PRP=f'{Path(__file__).resolve().parents[1]}/'

load_dotenv(dotenv_path=f"{PRP}/.env")
