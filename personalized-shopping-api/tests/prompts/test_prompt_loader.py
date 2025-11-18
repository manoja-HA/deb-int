"""
Tests for prompt loader

Verifies that:
- Prompts can be loaded from files
- Variables are correctly substituted
- Required variables are validated
- Metadata is correctly parsed
"""

import pytest
from pathlib import Path
from app.prompts.loader import PromptLoader


@pytest.fixture
def prompt_loader():
    """Create prompt loader with test prompts directory"""
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    return PromptLoader(prompts_dir)


def test_load_response_generation_prompt(prompt_loader):
    """Test loading response generation prompt"""
    prompt_data = prompt_loader.load_prompt("response.generation")

    # Check system prompt
    assert prompt_data.system is not None
    assert len(prompt_data.system) > 0
    assert "personalized shopping assistant" in prompt_data.system.lower()

    # Check user prompt template
    assert prompt_data.user is not None
    assert "{{customer_name}}" in prompt_data.user
    assert "{{recommendations_text}}" in prompt_data.user

    # Check metadata
    assert prompt_data.metadata.id == "response.generation"
    assert prompt_data.metadata.version == "1.0.0"
    assert "response" in prompt_data.metadata.tags


def test_load_intent_classification_prompt(prompt_loader):
    """Test loading intent classification prompt"""
    prompt_data = prompt_loader.load_prompt("intent.classification")

    # Check system prompt
    assert "intelligent query classifier" in prompt_data.system.lower()
    assert "INFORMATIONAL" in prompt_data.system
    assert "RECOMMENDATION" in prompt_data.system

    # Check metadata
    assert prompt_data.metadata.id == "intent.classification"
    assert prompt_data.metadata.temperature == 0.1  # Low for classification


def test_render_user_prompt_with_variables(prompt_loader):
    """Test rendering user prompt with variable substitution"""
    rendered = prompt_loader.render_user_prompt(
        "response.generation",
        customer_name="John Smith",
        price_segment="premium",
        favorite_categories="Electronics, Books",
        purchase_frequency="high",
        recommendations_text="1. Laptop ($800)\n2. Mouse ($30)",
    )

    # Verify variables were substituted
    assert "John Smith" in rendered
    assert "premium" in rendered
    assert "Electronics, Books" in rendered
    assert "Laptop" in rendered
    assert "{{" not in rendered  # No unrendered variables


def test_render_missing_required_variable_raises_error(prompt_loader):
    """Test that missing required variables raise ValueError"""
    with pytest.raises(ValueError) as exc_info:
        prompt_loader.render_user_prompt(
            "response.generation",
            customer_name="John",
            # Missing other required variables
        )

    assert "Missing required variables" in str(exc_info.value)


def test_get_system_prompt(prompt_loader):
    """Test getting system prompt without variables"""
    system_prompt = prompt_loader.get_system_prompt("response.generation")

    assert system_prompt is not None
    assert len(system_prompt) > 0
    assert "{{" not in system_prompt  # System prompts typically don't have variables


def test_get_metadata(prompt_loader):
    """Test getting prompt metadata"""
    metadata = prompt_loader.get_metadata("response.generation")

    assert metadata.id == "response.generation"
    assert metadata.version is not None
    assert metadata.model is not None
    assert metadata.temperature >= 0
    assert metadata.max_tokens > 0
    assert len(metadata.variables) > 0


def test_list_prompts(prompt_loader):
    """Test listing all available prompts"""
    prompts = prompt_loader.list_prompts()

    assert len(prompts) >= 2  # At least response.generation and intent.classification
    assert "response.generation" in prompts
    assert "intent.classification" in prompts

    # Check metadata
    for prompt_id, metadata in prompts.items():
        assert metadata.id == prompt_id
        assert metadata.version is not None


def test_cache_works(prompt_loader):
    """Test that prompt caching works"""
    # Load first time
    prompt1 = prompt_loader.load_prompt("response.generation")

    # Load second time (should be from cache)
    prompt2 = prompt_loader.load_prompt("response.generation")

    # Should be the same object
    assert prompt1 is prompt2


def test_clear_cache(prompt_loader):
    """Test clearing the prompt cache"""
    # Load prompt
    prompt_loader.load_prompt("response.generation")

    # Clear cache
    prompt_loader.clear_cache()

    # Load again (should reload from disk)
    prompt_loader.load_prompt("response.generation")


def test_reload_prompt(prompt_loader):
    """Test reloading a specific prompt"""
    # Load first time
    prompt1 = prompt_loader.load_prompt("response.generation")

    # Reload
    prompt2 = prompt_loader.reload_prompt("response.generation")

    # Should be different objects
    assert prompt1 is not prompt2
