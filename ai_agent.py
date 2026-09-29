"""
ai_agent.py — AI Agent backed by Azure OpenAI + Cosmos DB vector search.

This agent exposes the same interface as the original Groq/Tavily langgraph agent
(get_response_from_ai_agent) but is implemented with the **native OpenAI function-
calling** API so it has **no** dependency on the uninstalled langchain / langgraph /
groq / tavily packages.

Behaviour:
  * The agent is defined by a customizable system prompt.
  * When "allow_search" is enabled, the agent can call a `cosmos_vector_search`
    tool that retrieves relevant CFTC document chunks from Cosmos DB (reusing the
    RAG core). The retrieved snippets are fed back to the model for a grounded
    final answer.
  * The LLM is Azure OpenAI (from the shared `.env` config in rag_core).
"""

import json
import logging

from rag_core import (
    openai_client,
    openai_completions_deployment,
    cftc_container,
    generate_embeddings,
    vector_search,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Tool implementation: Cosmos DB vector search over the CFTC corpus
# ---------------------------------------------------------------------
COSMOS_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "cosmos_vector_search",
        "description": (
            "Search the CFTC legal-document knowledge base stored in Cosmos DB. "
            "Use this for any question about CFTC enforcement actions, legal "
            "documents, or content in the database. Returns matching document "
            "snippets with source links."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question or search phrase.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def _cosmos_vector_search(query: str, num_results: int = 5) -> str:
    """Search the CFTC document corpus in Cosmos DB and return formatted results.

    Each result includes the document title, a snippet, and the source blob link.
    """
    try:
        embedding = generate_embeddings(query)
        results = vector_search(cftc_container, embedding, num_results=num_results)
    except Exception as e:
        logger.error(f"Cosmos vector search failed: {e}", exc_info=True)
        return f"Cosmos vector search failed: {e}"

    if not results:
        return "No relevant documents found in the CFTC knowledge base."

    lines = []
    for r in results:
        doc = r.get("document", {})
        source = doc.get("source", "Unknown")
        content = (doc.get("content", "") or "")[:500]
        blob_url = doc.get("blob_url", "")
        source_path = doc.get("source_path", "")
        score = r.get("SimilarityScore", 0)
        header = f"**Source:** {source} (similarity {round(float(score), 4)})"
        if blob_url:
            header += f"\n[Open document]({blob_url})"
        elif source_path:
            header += f"\nPath: {source_path}"
        lines.append(header + "\n" + content)
    return "\n\n".join(lines)


def get_search_tool():
    """Return the Cosmos vector-search tool spec (OpenAI function schema).

    Kept for backward compatibility with tests that referenced a search tool.
    """
    return COSMOS_TOOL_SPEC


# ---------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------


class CosmosSearchResults:
    """A callable tool that performs Cosmos DB vector search."""

    name = COSMOS_TOOL_SPEC["function"]["name"]
    description = COSMOS_TOOL_SPEC["function"]["description"]

    def run(self, query, num_results=5):
        return _cosmos_vector_search(query, num_results)

    def invoke(self, query, num_results=5):
        return self.run(query, num_results)


def format_tool_response(tool_response):
    """Format a Cosmos search result string into readable markdown.

    Kept for backward compatibility with tests. The tool response is typically
    already a formatted string; if it is a list/dict, we stringify it.
    """
    if isinstance(tool_response, str):
        return tool_response.strip() or "No search results found."
    if isinstance(tool_response, dict):
        if isinstance(tool_response.get("results"), list):
            return format_tool_response(tool_response["results"])
        if not tool_response:
            return "No search results found."
        return str(tool_response.get("results", tool_response))
    if isinstance(tool_response, list):
        parts = []
        for item in tool_response:
            if isinstance(item, dict):
                title = item.get("title", item.get("name", ""))
                link = item.get("link", item.get("url", ""))
                snippet = item.get("snippet", item.get("content", ""))
                if title or snippet or link:
                    parts.append(
                        "**" + title + "**\n" + snippet + "\n[Read more](" + link + ")"
                    )
                    continue
            parts.append(str(item))
        return "\n\n".join(parts) if parts else "No search results found."
    return str(tool_response)


# `groq_llm` is kept as a sentinel/alias for backward compatibility. The real
# implementation uses Azure OpenAI (see get_response_from_ai_agent).
groq_llm = "azure-openai (via rag_core.openai_client)"


def _extract_final_answer(result):
    """Extract the final assistant answer text.

    Accepts either a plain string or a response dict with a ``messages`` key
    (e.g. the dict returned by ``get_response_from_ai_agent``). Returns the
    normalized assistant text.
    """
    # If given a response dict with messages, pull the last assistant content.
    if isinstance(result, dict):
        messages = result.get("messages") or []
        for msg in reversed(messages):
            if isinstance(msg, dict) and "content" in msg:
                return (msg.get("content") or "").strip()
        return ""
    return (result or "").strip()


def _run_tool_calls(tool_calls, messages):
    """Execute any tool calls requested by the model and append tool messages.

    Returns the updated messages list (ready for the follow-up model call).
    """
    for tc in tool_calls:
        fn = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if fn == "cosmos_vector_search":
            result = _cosmos_vector_search(
                args.get("query", ""), args.get("num_results", 5)
            )
        else:
            result = f"Unknown tool: {fn}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )
    return messages


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    """Query the Azure OpenAI agent (with optional Cosmos retrieval).

    Args:
        llm_id: Azure OpenAI deployment name (e.g. 'gpt-5-csl14'). If empty/None,
                the `.env` completion deployment is used.
        query:   User question (str, or list of str).
        allow_search: If True, enable the Cosmos DB retrieval tool.
        system_prompt: Agent system prompt.
        provider: 'azure' (Groq no longer available; 'groq' is accepted as alias).
    """
    try:
        logger.info(
            f"get_response_from_ai_agent: llm_id={llm_id}, "
            f"allow_search={allow_search}, provider={provider}"
        )

        if provider not in ("azure", "groq"):
            raise ValueError(f"Unsupported provider: {provider}")

        # Resolve user text
        user_text = query[0] if isinstance(query, list) else query
        user_text = user_text or ""

        # Resolve the deployment/model name
        model = llm_id or openai_completions_deployment

        # Build the message list
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        tools = [COSMOS_TOOL_SPEC] if allow_search else []

        # First completion
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = openai_client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        if not getattr(message, "tool_calls", None):
            final = _extract_final_answer(message.content)
            return {
                "messages": [{"role": "assistant", "content": final}],
                "raw": response,
            }

        # Tool calling loop (bounded to avoid infinite loops)
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )
        messages = _run_tool_calls(message.tool_calls, messages)

        for _ in range(4):
            response = openai_client.chat.completions.create(
                model=model, messages=messages
            )
            message = response.choices[0].message
            if not getattr(message, "tool_calls", None):
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
            messages = _run_tool_calls(message.tool_calls, messages)

        final = _extract_final_answer(message.content)
        return {
            "messages": [{"role": "assistant", "content": final}],
            "raw": response,
        }
    except Exception as e:
        logger.error(f"Error in get_response_from_ai_agent: {e}", exc_info=True)
        raise
