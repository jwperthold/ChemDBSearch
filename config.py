from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    smallworld_url: str = "https://sw.docking.org"
    enamine_api_key: Optional[str] = None  # For future direct API

    # Search Defaults
    default_similarity_threshold: float = 0.7  # Tanimoto threshold
    default_max_results: int = 100
    default_database: str = "REAL-Database-22Q1.smi.anon"

    # Output Configuration
    output_dir: Path = Path("./results")
    include_3d_coords: bool = False  # Whether to preserve 3D coords in output

    class Config:
        env_file = ".env"
        env_prefix = "CHEMDB_"


# Global settings instance
settings = Settings()
