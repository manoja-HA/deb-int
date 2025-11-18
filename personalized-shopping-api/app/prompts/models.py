"""
Pydantic models for prompt management
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import date


class ChangelogEntry(BaseModel):
    """Single changelog entry for a prompt version"""
    version: str
    date: date
    changes: str


class PromptExample(BaseModel):
    """Example input/output for a prompt"""
    query: str
    expected: Dict[str, Any]


class PromptMetadata(BaseModel):
    """Metadata for a prompt template"""
    id: str = Field(description="Unique prompt identifier")
    version: str = Field(description="Semantic version (e.g., '1.0.0')")
    description: str = Field(description="What this prompt does")
    model: str = Field(description="Default model to use")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, gt=0)

    variables: Dict[str, List[str]] = Field(
        description="Required variables for each prompt part"
    )
    output_format: str = Field(description="Output format (string, structured, etc.)")
    output_schema: Optional[str] = Field(
        default=None,
        description="Pydantic model name for structured output"
    )

    tags: List[str] = Field(default_factory=list)
    examples: Optional[List[PromptExample]] = Field(default=None)
    changelog: List[ChangelogEntry] = Field(default_factory=list)


class PromptData(BaseModel):
    """Complete prompt data including content and metadata"""
    system: str = Field(description="System prompt content")
    user: str = Field(description="User prompt template")
    metadata: PromptMetadata = Field(description="Prompt metadata")
