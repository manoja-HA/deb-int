# Agent Architecture Quick Start Guide

## Introduction

This guide shows you how to work with the new agentic architecture in 5 minutes.

## Core Concepts

1. **Agents**: Single-purpose capabilities that implement `BaseAgent`
2. **Workflows**: Orchestrate agents into complete use cases
3. **Services**: Thin facades that route to workflows
4. **No Business Logic in Workflows**: All logic lives in agents

---

## Creating Your First Agent

### Step 1: Define Input/Output Schemas

```python
from pydantic import BaseModel, Field

class MyAgentInput(BaseModel):
    """Input for my agent"""
    customer_id: str = Field(description="Customer to analyze")
    min_purchases: int = Field(default=5, ge=1)

class MyAgentOutput(BaseModel):
    """Output from my agent"""
    is_loyal: bool = Field(description="Whether customer is loyal")
    confidence: float = Field(ge=0, le=1)
```

### Step 2: Implement the Agent

```python
from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext

class LoyaltyAgent(BaseAgent[MyAgentInput, MyAgentOutput]):
    """Agent that determines customer loyalty"""

    def __init__(self, customer_repository):
        metadata = AgentMetadata(
            id="loyalty_detection",
            name="Loyalty Detection Agent",
            description="Determines if a customer is loyal based on purchase history",
            version="1.0.0",
            input_schema=MyAgentInput,
            output_schema=MyAgentOutput,
            tags=["customer", "loyalty", "analysis"],
        )
        super().__init__(metadata)
        self.customer_repo = customer_repository

    async def _execute(
        self,
        input_data: MyAgentInput,
        context: AgentContext,
    ) -> MyAgentOutput:
        # Your business logic here
        purchases = self.customer_repo.get_purchases_by_customer_id(
            input_data.customer_id
        )

        is_loyal = len(purchases) >= input_data.min_purchases
        confidence = min(len(purchases) / 20.0, 1.0)

        self._logger.info(
            f"Customer {input_data.customer_id}: "
            f"{'loyal' if is_loyal else 'not loyal'} "
            f"({len(purchases)} purchases)"
        )

        return MyAgentOutput(
            is_loyal=is_loyal,
            confidence=confidence,
        )
```

### Step 3: Use the Agent

```python
from app.repositories.customer_repository import CustomerRepository
from app.capabilities.base import AgentContext

# Create agent
customer_repo = CustomerRepository()
agent = LoyaltyAgent(customer_repo)

# Prepare input
input_data = MyAgentInput(
    customer_id="123",
    min_purchases=10
)

# Create context
context = AgentContext(
    request_id="req-abc123",
    user_id="123",
)

# Run agent (async)
output = await agent.run(input_data, context)

# Use result
if output.is_loyal:
    print(f"Customer is loyal (confidence: {output.confidence:.0%})")
```

---

## Using Existing Agents

### Example: Customer Profiling

```python
from app.capabilities.agents import CustomerProfilingAgent
from app.capabilities.agents.customer_profiling import CustomerProfilingInput

# Initialize agent
from app.repositories.customer_repository import CustomerRepository
agent = CustomerProfilingAgent(CustomerRepository())

# Prepare input
input_data = CustomerProfilingInput(customer_id="887")

# Run
output = await agent.run(input_data, context)

# Access profile
profile = output.profile
print(f"{profile.customer_name}: {profile.price_segment} segment")
print(f"Favorite categories: {profile.favorite_categories}")
```

### Example: Similar Customers

```python
from app.capabilities.agents import SimilarCustomersAgent
from app.capabilities.agents.similar_customers import SimilarCustomersInput

# Initialize agent
from app.repositories.vector_repository import VectorRepository
agent = SimilarCustomersAgent(
    customer_repository=CustomerRepository(),
    vector_repository=VectorRepository()
)

# Prepare input (requires customer profile)
input_data = SimilarCustomersInput(
    customer_profile=profile,  # From profiling agent
    top_k=20,
    similarity_threshold=0.75,
)

# Run
output = await agent.run(input_data, context)

# Access similar customers
for customer in output.similar_customers:
    print(f"{customer.customer_name}: {customer.similarity_score:.0%} similar")
    print(f"  Common categories: {customer.common_categories}")
```

### Example: Product Scoring

```python
from app.capabilities.agents import ProductScoringAgent
from app.capabilities.agents.product_scoring import (
    ProductScoringInput,
    ScoredProduct,
)

# Initialize agent (no dependencies)
agent = ProductScoringAgent()

# Prepare products
products = [
    ScoredProduct(
        product_id="1",
        product_name="Laptop",
        product_category="Electronics",
        avg_price=800.0,
        purchase_count=10,
        avg_sentiment=0.9,
        review_count=50,
    ),
    # ... more products
]

# Prepare input
input_data = ProductScoringInput(
    customer_profile=profile,
    products=products,
    purchase_counts={"1": 10, "2": 8},  # How many similar customers bought each
    top_n=5,
    max_per_category=2,
)

# Run
output = await agent.run(input_data, context)

# Access recommendations
for rec in output.recommendations:
    print(f"{rec.product_name}: {rec.recommendation_score:.2f}")
    print(f"  Reason: {rec.reason}")
```

---

## Composing Agents into Workflows

### Simple Workflow Example

```python
class SimpleWorkflow:
    """Workflow that uses 2 agents"""

    def __init__(self, customer_repo, vector_repo):
        self.profiling_agent = CustomerProfilingAgent(customer_repo)
        self.similar_agent = SimilarCustomersAgent(customer_repo, vector_repo)

    async def execute(self, customer_id: str, context: AgentContext):
        # Agent 1: Profile customer
        profiling_output = await self.profiling_agent.run(
            CustomerProfilingInput(customer_id=customer_id),
            context
        )

        # Agent 2: Find similar customers
        similar_output = await self.similar_agent.run(
            SimilarCustomersInput(
                customer_profile=profiling_output.profile,
                top_k=10,
            ),
            context
        )

        # Return result
        return {
            "profile": profiling_output.profile,
            "similar": similar_output.similar_customers,
        }
```

### Using the Workflow

```python
from app.repositories.customer_repository import CustomerRepository
from app.repositories.vector_repository import VectorRepository

# Initialize workflow
workflow = SimpleWorkflow(
    customer_repo=CustomerRepository(),
    vector_repo=VectorRepository(),
)

# Execute
context = AgentContext(request_id="req-123")
result = await workflow.execute("887", context)

# Access results
print(f"Customer: {result['profile'].customer_name}")
print(f"Similar customers: {len(result['similar'])}")
```

---

## Testing Agents

### Unit Test Example

```python
import pytest
from unittest.mock import Mock
from app.capabilities.base import AgentContext

@pytest.mark.asyncio
async def test_loyalty_agent():
    # Mock repository
    mock_repo = Mock()
    mock_repo.get_purchases_by_customer_id.return_value = [
        {"product_id": "1"},
        {"product_id": "2"},
        {"product_id": "3"},
    ]

    # Create agent with mock
    agent = LoyaltyAgent(mock_repo)

    # Prepare input
    input_data = MyAgentInput(customer_id="123", min_purchases=2)
    context = AgentContext(request_id="test")

    # Run agent
    output = await agent.run(input_data, context)

    # Assertions
    assert output.is_loyal is True  # 3 purchases >= 2 min
    assert 0 < output.confidence <= 1
    mock_repo.get_purchases_by_customer_id.assert_called_once_with("123")
```

---

## Agent Registry

### Register an Agent

```python
from app.capabilities.base import AgentRegistry

# Register for discovery
AgentRegistry.register(
    agent_class=LoyaltyAgent,
    metadata=agent.metadata,
)
```

### Discover Agents

```python
# List all agents
all_agents = AgentRegistry.list_agents()
for metadata in all_agents:
    print(f"{metadata.id}: {metadata.description}")

# Find by tag
customer_agents = AgentRegistry.find_by_tag("customer")
for metadata in customer_agents:
    print(f"  - {metadata.name}")

# Get specific agent
agent_class = AgentRegistry.get_agent("loyalty_detection")
agent_metadata = AgentRegistry.get_metadata("loyalty_detection")
```

---

## Observability

### Automatic Logging

All agents automatically log:
- Start/completion messages
- Execution time
- Errors (with stack traces)

```python
# Agent logs automatically:
# INFO: Agent 'loyalty_detection' starting (request_id=req-123)
# INFO: Agent 'loyalty_detection' completed successfully (execution_time_ms=15.2)
```

### Execution Metadata

Agent execution data is stored in context:

```python
output = await agent.run(input_data, context)

# Check execution metadata
executions = context.metadata.get("agent_executions", [])
for execution in executions:
    print(f"Agent: {execution['agent_id']}")
    print(f"Time: {execution['execution_time_ms']:.1f}ms")
    print(f"Success: {execution['success']}")
```

### Custom Logging

Add custom logs in your agent:

```python
async def _execute(self, input_data, context):
    self._logger.info(f"Processing customer {input_data.customer_id}")
    self._logger.debug(f"Min purchases: {input_data.min_purchases}")

    # ... logic ...

    if is_loyal:
        self._logger.info(f"Loyal customer detected!")

    return MyAgentOutput(...)
```

---

## Best Practices

### 1. Keep Agents Small

✅ **Good**: Single, focused responsibility
```python
class SentimentFilteringAgent:
    """Filters products by sentiment"""
    # Does ONE thing well
```

❌ **Bad**: Multiple responsibilities
```python
class ProductAgent:
    """Filters, scores, ranks, and formats products"""
    # Does too many things
```

### 2. Use Dependency Injection

✅ **Good**: Dependencies via constructor
```python
class MyAgent(BaseAgent):
    def __init__(self, repo: Repository, analyzer: Analyzer):
        # Inject dependencies
        self.repo = repo
        self.analyzer = analyzer
```

❌ **Bad**: Create dependencies internally
```python
class MyAgent(BaseAgent):
    def __init__(self):
        self.repo = Repository()  # Hard to test!
```

### 3. Keep Workflows Simple

✅ **Good**: Pure orchestration
```python
async def execute(self):
    output1 = await agent1.run(input1, ctx)
    output2 = await agent2.run(output1, ctx)
    return build_response(output2)
```

❌ **Bad**: Business logic in workflow
```python
async def execute(self):
    output1 = await agent1.run(input1, ctx)
    # Don't do this:
    if output1.score > 0.8:
        adjusted_score = output1.score * 1.2
    # Put logic in agent instead!
```

### 4. Validate Inputs

✅ **Good**: Use Pydantic constraints
```python
class MyInput(BaseModel):
    top_n: int = Field(ge=1, le=20)  # Between 1 and 20
    threshold: float = Field(ge=0.0, le=1.0)  # Between 0 and 1
```

### 5. Document Your Agents

✅ **Good**: Clear docstrings
```python
class MyAgent(BaseAgent):
    """
    Agent that does X

    Responsibilities:
    - Fetches data from Y
    - Calculates Z
    - Returns W

    This agent encapsulates the X logic that was previously
    embedded in Service ABC.
    """
```

---

## Common Patterns

### Pattern 1: Data Transformation

```python
# Agent transforms data from one format to another
class DataTransformAgent(BaseAgent):
    async def _execute(self, input_data, context):
        # Fetch raw data
        raw = self.repo.get_data(input_data.id)

        # Transform
        transformed = self._transform(raw)

        return TransformOutput(data=transformed)
```

### Pattern 2: Filtering

```python
# Agent filters a list based on criteria
class FilterAgent(BaseAgent):
    async def _execute(self, input_data, context):
        items = input_data.items
        threshold = input_data.threshold

        # Filter
        filtered = [item for item in items if item.score >= threshold]

        return FilterOutput(filtered_items=filtered)
```

### Pattern 3: Enrichment

```python
# Agent enriches data with additional information
class EnrichmentAgent(BaseAgent):
    async def _execute(self, input_data, context):
        items = input_data.items

        # Enrich each item
        enriched = []
        for item in items:
            metadata = self.repo.get_metadata(item.id)
            enriched.append(EnrichedItem(**item.dict(), metadata=metadata))

        return EnrichmentOutput(items=enriched)
```

---

## Troubleshooting

### Issue: Agent not running

**Problem**: `agent.run()` doesn't execute

**Solution**: Make sure you're using `await`:
```python
output = await agent.run(input_data, context)  # ✅
# not: output = agent.run(input_data, context)  # ❌
```

### Issue: Type errors

**Problem**: Pydantic validation fails

**Solution**: Check your input matches the schema:
```python
# Check schema
print(agent.get_input_schema().schema())

# Validate manually
input_data = MyInput.parse_obj({"field": "value"})
```

### Issue: Agent errors not logged

**Problem**: Errors silently swallowed

**Solution**: Agents automatically log errors. Check logs at `DEBUG` level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Next Steps

1. **Read Full Documentation**: [ARCHITECTURE_REFACTORING.md](./ARCHITECTURE_REFACTORING.md)
2. **See Examples**: Check `app/capabilities/agents/` for production agents
3. **Run Tests**: `pytest tests/capabilities/` to see test patterns
4. **Explore Workflows**: Look at `app/workflows/personalized_recommendation.py`

---

**Happy Agent Building! 🚀**
