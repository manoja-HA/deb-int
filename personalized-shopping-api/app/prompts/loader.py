"""
Prompt loader with templating and versioning support
"""

from pathlib import Path
from typing import Dict, Optional
import yaml
from jinja2 import Template, Environment, FileSystemLoader
import logging

from app.prompts.models import PromptMetadata, PromptData

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Centralized prompt loader with file-based templates

    Features:
    - Load prompts from file system
    - Jinja2 templating with variable substitution
    - Metadata and versioning support
    - Caching for performance
    - Validation of required variables

    Usage:
        loader = PromptLoader(prompts_dir=Path("prompts"))
        prompt_data = loader.load_prompt("response.generation")
        rendered = loader.render_user_prompt(
            "response.generation",
            customer_name="John",
            recommendations_text="1. Laptop..."
        )
    """

    def __init__(self, prompts_dir: Path):
        """
        Initialize prompt loader

        Args:
            prompts_dir: Base directory containing prompt templates
        """
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, PromptData] = {}
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
        )

    def load_prompt(self, prompt_id: str) -> PromptData:
        """
        Load prompt with metadata

        Args:
            prompt_id: Prompt identifier (e.g., "response.generation")

        Returns:
            Complete prompt data with content and metadata

        Raises:
            FileNotFoundError: If prompt files don't exist
            ValueError: If metadata is invalid
        """
        # Check cache
        if prompt_id in self._cache:
            return self._cache[prompt_id]

        # Parse ID to folder name (e.g., "response.generation" -> "response_generation")
        folder = prompt_id.replace(".", "_")
        prompt_folder = self.prompts_dir / folder

        if not prompt_folder.exists():
            raise FileNotFoundError(f"Prompt folder not found: {prompt_folder}")

        # Load system prompt
        system_file = prompt_folder / "system.txt"
        if not system_file.exists():
            raise FileNotFoundError(f"System prompt not found: {system_file}")
        system_content = system_file.read_text().strip()

        # Load user prompt
        user_file = prompt_folder / "user.txt"
        if not user_file.exists():
            raise FileNotFoundError(f"User prompt not found: {user_file}")
        user_content = user_file.read_text().strip()

        # Load metadata
        metadata_file = prompt_folder / "metadata.yaml"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata_dict = yaml.safe_load(f)

        metadata = PromptMetadata(**metadata_dict)

        # Create prompt data
        prompt_data = PromptData(
            system=system_content,
            user=user_content,
            metadata=metadata,
        )

        # Cache it
        self._cache[prompt_id] = prompt_data

        logger.debug(f"Loaded prompt: {prompt_id} (version {metadata.version})")

        return prompt_data

    def render_user_prompt(self, prompt_id: str, **variables) -> str:
        """
        Render user prompt with variables using Jinja2

        Args:
            prompt_id: Prompt identifier
            **variables: Template variables

        Returns:
            Rendered prompt text

        Raises:
            ValueError: If required variables are missing
        """
        prompt_data = self.load_prompt(prompt_id)

        # Validate required variables
        required_vars = prompt_data.metadata.variables.get("user", [])
        missing_vars = set(required_vars) - set(variables.keys())

        if missing_vars:
            raise ValueError(
                f"Missing required variables for {prompt_id}: {missing_vars}"
            )

        # Render template
        template = Template(prompt_data.user)
        rendered = template.render(**variables)

        return rendered

    def get_system_prompt(self, prompt_id: str) -> str:
        """
        Get system prompt (no variables typically)

        Args:
            prompt_id: Prompt identifier

        Returns:
            System prompt text
        """
        prompt_data = self.load_prompt(prompt_id)
        return prompt_data.system

    def get_metadata(self, prompt_id: str) -> PromptMetadata:
        """
        Get prompt metadata only

        Args:
            prompt_id: Prompt identifier

        Returns:
            Prompt metadata
        """
        prompt_data = self.load_prompt(prompt_id)
        return prompt_data.metadata

    def list_prompts(self) -> Dict[str, PromptMetadata]:
        """
        List all available prompts

        Returns:
            Dictionary of prompt_id -> metadata
        """
        prompts = {}

        for folder in self.prompts_dir.iterdir():
            if not folder.is_dir():
                continue

            # Convert folder name to prompt ID
            prompt_id = folder.name.replace("_", ".")

            try:
                metadata = self.get_metadata(prompt_id)
                prompts[prompt_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to load prompt {prompt_id}: {e}")

        return prompts

    def clear_cache(self):
        """Clear the prompt cache"""
        self._cache.clear()
        logger.info("Cleared prompt cache")

    def reload_prompt(self, prompt_id: str) -> PromptData:
        """
        Reload a specific prompt from disk

        Args:
            prompt_id: Prompt identifier

        Returns:
            Reloaded prompt data
        """
        if prompt_id in self._cache:
            del self._cache[prompt_id]

        return self.load_prompt(prompt_id)


# Singleton instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """
    Get singleton prompt loader instance

    Returns:
        Global PromptLoader instance
    """
    global _prompt_loader

    if _prompt_loader is None:
        from app.core.config import settings
        prompts_dir = settings.BASE_DIR / "prompts"

        if not prompts_dir.exists():
            raise RuntimeError(f"Prompts directory not found: {prompts_dir}")

        _prompt_loader = PromptLoader(prompts_dir)
        logger.info(f"Initialized PromptLoader with directory: {prompts_dir}")

    return _prompt_loader
