import asyncio
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

llm_config = LLMConfig(
    api_key="ollama",
    model="gemma4:26b",
    small_model="gemma4:26b",
    base_url="http://localhost:11434/v1",
)

llm_client = OpenAIGenericClient(config=llm_config)

embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
    api_key="ollama",
    embedding_model="nomic-embed-text",
    embedding_dim=768,
    base_url="http://localhost:11434/v1",
))

cross_encoder = OpenAIRerankerClient(
    client=llm_client,
    config=llm_config
)

graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=cross_encoder,
)

async def main():
    await graphiti.build_indices_and_constraints()

    await graphiti.add_episode(
        name="episode-1",
        episode_body="Alice is a software engineer who works at Acme Corp. She is friends with Bob who is a data scientist at the same company. Bob recently moved to Singapore.",
        source_description="test input",
        reference_time=datetime.now(timezone.utc),
    )

    print("Done! Go check Neo4j browser")
    await graphiti.close()

asyncio.run(main())