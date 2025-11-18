# Prompt Management in the Agentic Architecture

## Overview

Currently, prompts in the personalized shopping API are **stored inline within agent code** as string literals. This document explains the current approach, its limitations, and provides recommendations for improvement.

---

## Current State: Inline Prompts

### 1. Response Generation Agent

**Location**: `app/capabilities/agents/response_generation.py:122-130`

**How It's Stored**:
```python
# Inline f-string prompt
prompt = f"""Based on the customer's purchase history, provide a brief explanation for these recommendations.

{customer_context}

Recommendations:
{rec_text}

Provide a 2-3 sentence explanation focusing on why these products match the customer's preferences."""
```

**Characteristics**:
- ✅ Simple and easy to modify
- ✅ Template variables directly embedded (`{customer_context}`, `{rec_text}`)
- ❌ Hard to version or A/B test
- ❌ Not easily translatable to other languages
- ❌ Changes require code deployment

---

### 2. Intent Classification Agent

**Location**: `app/agents/intent_classifier_agent.py:127-161`

**How It's Stored**:
```python
system_prompt = """You are an intelligent query classifier for a personalized shopping assistant API.

Your task is to analyze user queries and classify them into two main categories:

1. **INFORMATIONAL** - Questions asking about customer data:
   - Total purchases (e.g., "how many items has X bought?")
   - Spending (e.g., "how much has X spent?", "what's the total amount?")
   - Favorite categories (e.g., "what categories does X like?")
   - Recent purchases (e.g., "what did X buy recently?")
   - Customer profile (e.g., "tell me about X", "show X's profile")
   - General questions about customer data

2. **RECOMMENDATION** - Requests for product suggestions:
   - Direct requests for recommendations
   - Questions about what to buy
   - Requests for product suggestions
   - Questions about what customer would like

Analyze the query and respond with a JSON object with these fields:
- intent: "informational" or "recommendation"
- category: if informational, one of: "total_purchases", "spending", "favorite_categories", "recent_purchases", "customer_profile", "general"
- confidence: a number between 0 and 1
- reasoning: brief explanation of your classification

Examples:
Query: "what is the total purchase of Kenneth Martinez?"
Response: {"intent": "informational", "category": "total_purchases", "confidence": 0.95, "reasoning": "Query explicitly asks about total purchases"}

Query: "recommend some products for Kenneth Martinez"
Response: {"intent": "recommendation", "category": null, "confidence": 0.98, "reasoning": "Direct request for product recommendations"}

Query: "how much money has John spent?"
Response: {"intent": "informational", "category": "spending", "confidence": 0.92, "reasoning": "Query asks about spending amount"}

Now classify the following query. Respond ONLY with valid JSON, no additional text."""
```

**Characteristics**:
- ✅ Comprehensive instructions
- ✅ Includes examples (few-shot prompting)
- ✅ Clear output format specification
- ❌ Very long (35+ lines)
- ❌ Hard to maintain in code
- ❌ Cannot be easily tested or versioned

---

## Prompt Inventory

| Agent | Prompt Type | Location | Size | Variables |
|-------|-------------|----------|------|-----------|
| **ResponseGenerationAgent** | User prompt | `response_generation.py:123` | ~7 lines | `customer_context`, `rec_text` |
| **IntentClassifierAgent** | System prompt | `intent_classifier_agent.py:127` | ~35 lines | None (static) |
| **IntentClassifierAgent** | User prompt | `intent_classifier_agent.py:163` | 1 line | `query` |

**Total Active Prompts**: 3
**Total Agents Using LLMs**: 2

---

## Prompt Template Patterns

### Pattern 1: Simple F-String Template

**Used by**: ResponseGenerationAgent

```python
# Variables injected directly into f-string
prompt = f"""Instruction text here.

{variable1}

More instructions.
{variable2}

Final instruction."""
```

**Pros**:
- Simple and readable
- Direct variable substitution
- No external dependencies

**Cons**:
- Hard to version
- Cannot be edited without code changes
- No validation

---

### Pattern 2: Multi-Part Prompts (System + User)

**Used by**: IntentClassifierAgent

```python
# System message (static instructions)
system_prompt = """You are an AI assistant...
Instructions here..."""

# User message (dynamic query)
user_prompt = f"Query: {query}"

# Combined in LLM call
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
]
```

**Pros**:
- Separates static instructions from dynamic content
- Follows OpenAI/LangChain message pattern
- Better context management

**Cons**:
- Still inline strings
- No centralized management

---

## Current Limitations

### 1. **No Centralized Prompt Registry**
- Prompts scattered across multiple files
- Hard to get an overview of all prompts
- Difficult to ensure consistency

### 2. **No Version Control for Prompts**
- Changes to prompts not tracked separately
- Cannot rollback to previous prompt versions
- No A/B testing capability

### 3. **No Prompt Templates Library**
- Cannot reuse common prompt patterns
- Each agent reinvents prompt structure
- No standardization

### 4. **No Prompt Validation**
- No checks for required variables
- No validation of output format
- Runtime errors if variables missing

### 5. **Hard to Internationalize**
- Prompts in English only
- Cannot easily translate
- No localization support

### 6. **Limited Observability**
- Prompts not logged separately in traces
- Hard to analyze which prompts work best
- Cannot easily debug prompt issues

---

## Recommended Improvements

### Option 1: Centralized Prompt Registry (Simple)

**Implementation**:

```python
# app/prompts/registry.py
from typing import Dict
from string import Template

class PromptRegistry:
    """Central registry for all LLM prompts"""

    PROMPTS = {
        # Response generation
        "response.generation.user": Template("""Based on the customer's purchase history, provide a brief explanation for these recommendations.

$customer_context

Recommendations:
$rec_text

Provide a 2-3 sentence explanation focusing on why these products match the customer's preferences."""),

        # Intent classification
        "intent.classification.system": """You are an intelligent query classifier for a personalized shopping assistant API.

Your task is to analyze user queries and classify them into two main categories:

1. **INFORMATIONAL** - Questions asking about customer data:
   - Total purchases (e.g., "how many items has X bought?")
   - Spending (e.g., "how much has X spent?", "what's the total amount?")
   - Favorite categories (e.g., "what categories does X like?")
   - Recent purchases (e.g., "what did X buy recently?")
   - Customer profile (e.g., "tell me about X", "show X's profile")

2. **RECOMMENDATION** - Requests for product suggestions:
   - Direct requests for recommendations
   - Questions about what to buy

Analyze the query and respond with a JSON object with these fields:
- intent: "informational" or "recommendation"
- category: if informational, specify category
- confidence: a number between 0 and 1
- reasoning: brief explanation

Respond ONLY with valid JSON, no additional text.""",

        "intent.classification.user": Template("Query: $query"),
    }

    @classmethod
    def get_prompt(cls, key: str, **variables) -> str:
        """Get prompt by key and substitute variables"""
        prompt_template = cls.PROMPTS.get(key)

        if prompt_template is None:
            raise ValueError(f"Prompt not found: {key}")

        if isinstance(prompt_template, Template):
            return prompt_template.safe_substitute(**variables)
        else:
            return prompt_template

    @classmethod
    def list_prompts(cls) -> Dict[str, str]:
        """List all available prompts"""
        return {k: str(v) for k, v in cls.PROMPTS.items()}
```

**Usage in Agent**:

```python
# Before
prompt = f"""Based on the customer's purchase history...
{customer_context}
...
{rec_text}
..."""

# After
from app.prompts.registry import PromptRegistry

prompt = PromptRegistry.get_prompt(
    "response.generation.user",
    customer_context=customer_context,
    rec_text=rec_text,
)
```

**Benefits**:
- ✅ All prompts in one place
- ✅ Easy to review and update
- ✅ Validates variable substitution
- ✅ Simple to implement

**Limitations**:
- Still in code (requires deployment)
- No version history
- No A/B testing

---

### Option 2: Database-Backed Prompts (Advanced)

**Implementation**:

```python
# Database schema
class Prompt(BaseModel):
    id: str
    key: str
    version: int
    content: str
    variables: List[str]
    created_at: datetime
    is_active: bool
    metadata: Dict[str, Any]

# Prompt manager
class PromptManager:
    def __init__(self, db_connection):
        self.db = db_connection

    def get_prompt(self, key: str, version: Optional[int] = None) -> str:
        """Get active or specific version of prompt"""
        if version:
            return self.db.query(Prompt).filter(
                Prompt.key == key,
                Prompt.version == version
            ).first()
        else:
            return self.db.query(Prompt).filter(
                Prompt.key == key,
                Prompt.is_active == True
            ).first()

    def create_version(self, key: str, content: str) -> Prompt:
        """Create new prompt version"""
        latest = self.get_latest_version(key)
        new_version = latest.version + 1 if latest else 1

        return Prompt(
            key=key,
            version=new_version,
            content=content,
            is_active=False,
        )

    def activate_version(self, key: str, version: int):
        """Activate specific prompt version"""
        # Deactivate all versions
        self.db.query(Prompt).filter(Prompt.key == key).update(
            {"is_active": False}
        )
        # Activate specific version
        self.db.query(Prompt).filter(
            Prompt.key == key,
            Prompt.version == version
        ).update({"is_active": True})
```

**Benefits**:
- ✅ Version history
- ✅ Runtime updates (no deployment)
- ✅ A/B testing support
- ✅ Audit trail

**Limitations**:
- More complex to implement
- Requires database
- Performance overhead

---

### Option 3: File-Based Prompt Templates (Recommended)

**Implementation**:

```
prompts/
├── response_generation/
│   ├── user.txt
│   └── metadata.json
├── intent_classification/
│   ├── system.txt
│   ├── user.txt
│   └── metadata.json
└── README.md
```

**Example Files**:

`prompts/response_generation/user.txt`:
```
Based on the customer's purchase history, provide a brief explanation for these recommendations.

{{customer_context}}

Recommendations:
{{rec_text}}

Provide a 2-3 sentence explanation focusing on why these products match the customer's preferences.
```

`prompts/response_generation/metadata.json`:
```json
{
  "id": "response.generation.user",
  "version": "1.0.0",
  "variables": ["customer_context", "rec_text"],
  "model": "llama3.1:8b",
  "temperature": 0.1,
  "max_tokens": 2048,
  "tags": ["response", "explanation", "reasoning"]
}
```

**Loader**:

```python
# app/prompts/loader.py
import json
from pathlib import Path
from typing import Dict, Any
from jinja2 import Template

class PromptLoader:
    """Load prompts from files"""

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """Load prompt with metadata"""
        if prompt_id in self._cache:
            return self._cache[prompt_id]

        # Parse ID (e.g., "response.generation.user" -> response_generation/user.txt)
        parts = prompt_id.split(".")
        folder = "_".join(parts[:-1])
        file = parts[-1]

        prompt_path = self.prompts_dir / folder / f"{file}.txt"
        metadata_path = self.prompts_dir / folder / "metadata.json"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        # Load prompt content
        content = prompt_path.read_text()

        # Load metadata
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        result = {
            "content": content,
            "metadata": metadata,
        }

        self._cache[prompt_id] = result
        return result

    def render_prompt(self, prompt_id: str, **variables) -> str:
        """Render prompt with variables using Jinja2"""
        prompt_data = self.load_prompt(prompt_id)
        template = Template(prompt_data["content"])
        return template.render(**variables)
```

**Usage**:

```python
# In agent
from app.prompts.loader import PromptLoader
from app.core.config import settings

prompt_loader = PromptLoader(settings.PROMPTS_DIR)

# Render prompt
prompt = prompt_loader.render_prompt(
    "response.generation.user",
    customer_context=customer_context,
    rec_text=rec_text,
)
```

**Benefits**:
- ✅ Prompts separate from code
- ✅ Version control via Git
- ✅ Easy to edit and review
- ✅ Metadata for configuration
- ✅ Jinja2 templating power
- ✅ No database required
- ✅ Can be versioned in Git

**This is the recommended approach!**

---

## Migration Plan

### Phase 1: Extract Prompts to Registry (Quick Win)
1. Create `app/prompts/registry.py`
2. Move all inline prompts to registry
3. Update agents to use `PromptRegistry.get_prompt()`
4. Test that functionality is unchanged

**Effort**: 2-3 hours
**Benefit**: Centralized prompts, easier to review

### Phase 2: Move to File-Based Templates (Recommended)
1. Create `prompts/` directory structure
2. Create prompt files (`.txt`) and metadata (`.json`)
3. Implement `PromptLoader` class
4. Update agents to use `PromptLoader.render_prompt()`
5. Add prompt validation tests

**Effort**: 4-6 hours
**Benefit**: Version control, easy editing, better organization

### Phase 3: Add Observability (Optional)
1. Log prompts in tracing spans
2. Track prompt versions used per request
3. Add metrics for prompt effectiveness
4. A/B testing infrastructure

**Effort**: 8-10 hours
**Benefit**: Better debugging, optimization

---

## Best Practices for Prompts

### 1. Use Clear Instructions
```
# ❌ Vague
"Explain the recommendations"

# ✅ Clear
"Provide a 2-3 sentence explanation focusing on why these products match the customer's preferences."
```

### 2. Include Examples (Few-Shot)
```
Examples:
Query: "what is the total purchase of Kenneth Martinez?"
Response: {"intent": "informational", "category": "total_purchases", ...}

Query: "recommend some products"
Response: {"intent": "recommendation", ...}
```

### 3. Specify Output Format
```
"Respond with a JSON object with these fields:
- intent: string
- confidence: number between 0 and 1
- reasoning: string

Respond ONLY with valid JSON, no additional text."
```

### 4. Use System/User Message Separation
```python
# System: Static instructions
system_prompt = "You are a helpful assistant..."

# User: Dynamic query
user_prompt = f"Customer query: {query}"
```

### 5. Version Prompts
```
# In metadata.json
{
  "version": "1.2.0",
  "changelog": [
    {"version": "1.0.0", "changes": "Initial prompt"},
    {"version": "1.1.0", "changes": "Added examples"},
    {"version": "1.2.0", "changes": "Improved output format"}
  ]
}
```

---

## Prompt Testing

### Unit Tests for Prompts

```python
# tests/prompts/test_prompts.py
import pytest
from app.prompts.loader import PromptLoader

def test_response_generation_prompt():
    loader = PromptLoader(Path("prompts"))

    prompt = loader.render_prompt(
        "response.generation.user",
        customer_context="Customer: John, mid-range buyer",
        rec_text="1. Laptop ($800)\n2. Mouse ($30)",
    )

    # Verify variables were substituted
    assert "John" in prompt
    assert "Laptop" in prompt
    assert "$800" in prompt

    # Verify prompt contains key instructions
    assert "customer's purchase history" in prompt
    assert "2-3 sentence explanation" in prompt


def test_intent_classification_prompt():
    loader = PromptLoader(Path("prompts"))

    prompt = loader.render_prompt(
        "intent.classification.user",
        query="How much has Kenneth Martinez spent?",
    )

    assert "How much has Kenneth Martinez spent?" in prompt


def test_missing_variable_raises_error():
    loader = PromptLoader(Path("prompts"))

    with pytest.raises(Exception):
        # Missing required variable
        loader.render_prompt("response.generation.user")
```

---

## Summary

### Current State
- ✅ **Prompts**: Inline strings in agent code
- ❌ **Management**: Scattered, no centralization
- ❌ **Versioning**: No version control for prompts
- ❌ **Testing**: No prompt-specific tests
- ❌ **Observability**: Limited prompt tracking

### Recommended Next Steps

1. **Short-term** (1-2 days):
   - Create `app/prompts/registry.py`
   - Move prompts to centralized registry
   - Update agents to use registry

2. **Medium-term** (1 week):
   - Implement file-based prompt templates
   - Create `prompts/` directory with `.txt` files
   - Add `PromptLoader` with Jinja2 support
   - Write prompt unit tests

3. **Long-term** (2-4 weeks):
   - Add prompt version tracking
   - Implement A/B testing infrastructure
   - Add prompt metrics and analytics
   - Create prompt engineering guidelines

### Benefits of Improvement

- ✅ **Maintainability**: Prompts easy to find and update
- ✅ **Collaboration**: Non-developers can edit prompts
- ✅ **Version Control**: Track prompt changes in Git
- ✅ **Testing**: Validate prompts independently
- ✅ **Optimization**: A/B test different prompts
- ✅ **Observability**: Track which prompts work best

---

**Ready to improve prompt management? Start with Phase 1! 🚀**
