from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from llama_stack_client import LlamaStackClient
import os
import asyncio
import json
import logging
import threading
import queue
import yaml
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import random

app = FastAPI(title="Canopy Backend API")

# Load configuration with fallback for testing
config_path = os.getenv("CANOPY_CONFIG_PATH", "/canopy/canopy-config.yaml")
# config_path = os.getenv("CANOPY_CONFIG_PATH", "./canopy-config.yaml")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llama_client = LlamaStackClient(base_url=config["LLAMA_STACK_URL"])

    # Feature flags configuration from environment variables
    FEATURE_FLAGS = {
        "information-search": "information-search" in config and config["information-search"]["enabled"] == True,
        "summarize": "summarize" in config and config["summarize"]["enabled"] == True,
        "student-assistant": "student-assistant" in config and config["student-assistant"]["enabled"] == True,
        "shields": "shields" in config and config["shields"]["enabled"] == True,
        "feedback": "feedback" in config and config["feedback"]["enabled"] == True,
    }

    # Shields configuration
    SHIELDS_CONFIG = {}
    if FEATURE_FLAGS.get("shields", False):
        SHIELDS_CONFIG = {
            "input_shields": config["shields"].get("input_shields", []),
            "output_shields": config["shields"].get("output_shields", []),
            "model": config["shields"].get("model", config.get("student-assistant", {}).get("model", "llama32")),
            "check_interval": config["shields"].get("check_interval", 50),
        }
else:
    # Fallback for testing - config will be loaded by tests
    config = None
    llama_client = None
    FEATURE_FLAGS = {
        "information-search": False,
        "summarize": False,
        "student-assistant": False,
        "shields": False,
        "feedback": False,
    }
    SHIELDS_CONFIG = {}

# Tool factory function for creating student assistant tools
# This allows tools to be imported and tested independently
def create_student_tools(llama_stack_client: LlamaStackClient, vector_store_id: str):
    """
    Factory function to create student assistant tools.

    This function is used both by the agent in production and by tests.
    It ensures a single source of truth for tool implementations.

    Args:
        llama_stack_client: LlamaStackClient instance for vector search
        vector_store_id: ID of the vector store to search

    Returns:
        List of LangChain tools (search_knowledge_base, find_professors_by_expertise)
    """
    # Professor directory - used by find_professors_by_expertise tool
    PROFESSORS = {
        "Dr. Sarah Chen": {
            "department": "Computer Science",
            "expertise": ["Machine Learning", "Neural Networks", "AI Ethics", "Agentic Workflows"],
            "email": "s.chen@university.edu"
        },
        "Prof. Michael Rodriguez": {
            "department": "Physics",
            "expertise": ["Quantum Mechanics", "Particle Physics", "Quantum Chromodynamics"],
            "email": "m.rodriguez@university.edu"
        },
        "Dr. Emily Thompson": {
            "department": "Biology",
            "expertise": ["Botany", "Ecology", "Forest Canopy Structure", "Plant Biology"],
            "email": "e.thompson@university.edu"
        },
        "Prof. James Wilson": {
            "department": "Computer Science",
            "expertise": ["Distributed Systems", "Cloud Computing", "Software Architecture"],
            "email": "j.wilson@university.edu"
        }
    }

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search through documents to find information. Use this when the user asks about concepts, definitions, or topics."""
        try:
            results = llama_stack_client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query,
                max_num_results=3,
                search_mode="vector",
                ranking_options={"score_threshold": 0.0}
            )
            if not results.data:
                return "No relevant information found in the knowledge base."
            formatted_results = []
            for i, result in enumerate(results.data, 1):
                content = result.content[0].text if hasattr(result, 'content') else str(result)
                formatted_results.append(f"Result {i}: {content}")
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    @tool
    def find_professors_by_expertise(topic: str) -> str:
        """Find professors who have expertise in a specific topic or subject area."""
        matching_profs = []
        for name, info in PROFESSORS.items():
            if any(topic.lower() in exp.lower() or exp.lower() in topic.lower() for exp in info["expertise"]):
                matching_profs.append((name, info))

        if not matching_profs:
            result = f"No professors found with specific expertise in '{topic}'.\n\nAvailable professors:\n\n"
            for name, info in PROFESSORS.items():
                result += f"**{name}** - {info['department']}\n  Expertise: {', '.join(info['expertise'])}\n  Email: {info['email']}\n\n"
            return result

        result = f"Professors with expertise in '{topic}':\n\n"
        for name, info in matching_profs:
            result += f"**{name}** - {info['department']}\n  Expertise: {', '.join(info['expertise'])}\n  Email: {info['email']}\n\n"
        return result

    return [search_knowledge_base, find_professors_by_expertise]


# Initialize student assistant agent if enabled
agent = None
if FEATURE_FLAGS.get("student-assistant", False):
    from datetime import datetime
    vector_store_id = config["student-assistant"].get("vector_db_id", "latest")

    # Create tools using factory function
    student_tools = create_student_tools(llama_client, vector_store_id)

    tools = student_tools + [
        {
            "type": "mcp",
            "server_label": "canopy-calendar",
            "server_url": config["student-assistant"].get("mcp_calendar_url", "http://canopy-mcp-calendar-mcp-server:8080/sse"),
            "require_approval": "never",
        }
    ]

    llm = ChatOpenAI(
        openai_api_base=config["LLAMA_STACK_URL"] + "/v1",
        model=config["student-assistant"]["model"],
        openai_api_key="not-needed",
        use_responses_api=True,
        temperature=config["student-assistant"].get("temperature", 0.1)
    )

    # Evaluate the prompt as an f-string to execute any Python expressions in {}
    prompt_template = config["student-assistant"].get("prompt", "You are a helpful university assistant.")
    formatted_prompt = eval(f'f"""{prompt_template}"""')

    agent = create_react_agent(
        llm,
        tools,
        prompt=formatted_prompt,
        checkpointer=MemorySaver()
    )

# Structured logger for feedback events (collected by LokiStack via STDOUT)
feedback_logger = logging.getLogger("canopy.feedback")
feedback_logger.setLevel(logging.INFO)
feedback_logger.propagate = False  # Prevent duplicate output from root logger
_feedback_handler = logging.StreamHandler()
_feedback_handler.setFormatter(logging.Formatter('%(message)s'))
feedback_logger.addHandler(_feedback_handler)

# In-memory feedback store (data lost on restart - acceptable for enablement)
feedback_store: List[Dict[str, Any]] = []

class PromptRequest(BaseModel):
    prompt: str

class FeedbackRequest(BaseModel):
    input_text: str
    response_text: str
    rating: str  # "thumbs_up" or "thumbs_down"
    feature: str = "summarize"
    comment: Optional[str] = None

@app.get("/feature-flags")
async def get_feature_flags() -> Dict[str, Any]:
    """Get all feature flags configuration"""
    return FEATURE_FLAGS

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit thumbs up/down feedback on an AI response."""
    if not FEATURE_FLAGS.get("feedback", False):
        raise HTTPException(status_code=404, detail="Feedback feature is not enabled")

    feedback_entry = {
        "id": len(feedback_store) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature": request.feature,
        "rating": request.rating,
        "input_text": request.input_text,
        "response_text": request.response_text,
        "comment": request.comment,
    }

    feedback_store.append(feedback_entry)

    # Structured log for LokiStack (JSON on STDOUT is auto-collected)
    feedback_logger.info(json.dumps({
        "event": "feedback_submitted",
        "feedback_id": feedback_entry["id"],
        "feature": request.feature,
        "rating": request.rating,
        "input_length": len(request.input_text),
        "response_length": len(request.response_text),
        "timestamp": feedback_entry["timestamp"],
    }))

    return {"status": "ok", "feedback_id": feedback_entry["id"]}


@app.get("/feedback")
async def list_feedback():
    """List all collected feedback entries."""
    if not FEATURE_FLAGS.get("feedback", False):
        raise HTTPException(status_code=404, detail="Feedback feature is not enabled")

    return {
        "total": len(feedback_store),
        "feedback": feedback_store,
    }


@app.get("/feedback/export")
async def export_feedback_for_eval():
    """Export negative feedback as eval-compatible dataset.

    Format matches canopy-evals summary_tests.yaml structure.
    """
    if not FEATURE_FLAGS.get("feedback", False):
        raise HTTPException(status_code=404, detail="Feedback feature is not enabled")

    negative_feedback = [
        entry for entry in feedback_store if entry["rating"] == "thumbs_down"
    ]

    tests = []
    for entry in negative_feedback:
        tests.append({
            "prompt": entry["input_text"],
            "expected_result": "",  # To be filled by human reviewer
        })

    eval_dataset = {
        "name": "feedback_eval_tests",
        "description": "Tests generated from negative user feedback on summarization.",
        "endpoint": "/summarize",
        "scoring_params": {
            "llm-as-judge::base": {
                "judge_model": "llama32",
                "prompt_template": "judge_prompt.txt",
                "type": "llm_as_judge",
                "judge_score_regexes": ["Answer: (A|B|C|D|E)"],
            },
            "basic::subset_of": None,
        },
        "tests": tests,
    }

    yaml_output = yaml.dump(eval_dataset, default_flow_style=False, allow_unicode=True, sort_keys=False)

    feedback_logger.info(json.dumps({
        "event": "feedback_exported",
        "total_negative": len(negative_feedback),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }))

    return Response(
        content=yaml_output,
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=feedback-eval-dataset.yaml"},
    )


@app.post("/summarize")
async def summarize(request: PromptRequest):
    # Check if summarization feature is enabled
    if not FEATURE_FLAGS.get("summarize", False):
        raise HTTPException(status_code=404, detail="Summarization feature is not enabled")

    sys_prompt = config["summarize"].get("prompt", "Summarize the following text:")
    temperature = config["summarize"].get("temperature", 0.7)
    max_tokens = config["summarize"].get("max_tokens", 4096)
    model = config["summarize"]["model"]

    q = queue.Queue()

    def worker_with_shields():
        """Use Responses API when shields are enabled - guardrails handled server-side."""
        print(f"sending request to model {model} with shields (Responses API)")
        try:
            input_shields = SHIELDS_CONFIG.get("input_shields", [])
            output_shields = SHIELDS_CONFIG.get("output_shields", [])
            # Combine input and output shields for guardrails
            guardrails = list(set(input_shields + output_shields))

            # Use Responses API with guardrails
            response = llama_client.responses.create(
                model=model,
                instructions=sys_prompt,
                input=[{"role": "user", "content": request.prompt, "type": "message"}],
                temperature=temperature,
                stream=True,
                extra_body={"guardrails": guardrails} if guardrails else None,
            )

            for event in response:
                # Handle different event types
                event_type = getattr(event, 'type', None)
                print(f"[Responses API] Event type: {event_type}")
                print(f"[Responses API] Event: {event}")

                # Handle text delta streaming
                if event_type == 'response.output_text.delta':
                    delta_text = getattr(event, 'delta', '')
                    if delta_text:
                        print(f"[Responses API] Delta: {delta_text}")
                        chunk = f"data: {json.dumps({'delta': delta_text})}\n\n"
                        q.put(chunk)

                # Handle response failure (includes guardrail violations)
                elif event_type == 'response.failed':
                    error_msg = 'Response generation failed'
                    if hasattr(event, 'response') and hasattr(event.response, 'error'):
                        error_msg = getattr(event.response.error, 'message', error_msg)
                    print(f"[Responses API] Failed: {error_msg}")
                    chunk = f"data: {json.dumps({'type': 'shield_violation', 'message': error_msg})}\n\n"
                    q.put(chunk)
                    break

                # Handle completed response
                elif event_type == 'response.completed':
                    print(f"[Responses API] Completed")
                    if hasattr(event, 'response'):
                        resp = event.response
                        # Check for refusal in output (guardrail violation)
                        if hasattr(resp, 'output') and resp.output:
                            for output_msg in resp.output:
                                if hasattr(output_msg, 'content') and output_msg.content:
                                    for content_item in output_msg.content:
                                        if isinstance(content_item, dict) and content_item.get('type') == 'refusal':
                                            error_msg = "Your request was blocked by our safety guardrails"
                                            print(f"[Responses API] Guardrail refusal detected")
                                            chunk = f"data: {json.dumps({'type': 'shield_violation', 'message': error_msg})}\n\n"
                                            q.put(chunk)
                                            return
                        # Check if response has error status
                        if hasattr(resp, 'status') and resp.status == 'failed':
                            error_msg = 'Content blocked by safety guardrails'
                            if hasattr(resp, 'error') and resp.error:
                                error_msg = getattr(resp.error, 'message', error_msg)
                            print(f"[Responses API] Guardrail violation: {error_msg}")
                            chunk = f"data: {json.dumps({'type': 'shield_violation', 'message': error_msg})}\n\n"
                            q.put(chunk)

        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)

    def worker_without_shields():
        """Use direct inference API when shields are disabled."""
        print(f"sending request to model {model} without shields (Inference API)")
        try:
            response = llama_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": request.prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    if delta.content:
                        chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
                        q.put(chunk)

        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)

    # Choose worker based on shields feature flag
    if FEATURE_FLAGS.get("shields", False):
        threading.Thread(target=worker_with_shields).start()
    else:
        threading.Thread(target=worker_without_shields).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.post("/information-search")
async def information_search(request: PromptRequest):
    # Check if information search feature is enabled
    if not FEATURE_FLAGS.get("information-search", False):
        raise HTTPException(status_code=404, detail="Information search feature is not enabled")

    # Dummy information search implementation
    sys_prompt = config["information-search"]["prompt"]
    temperature = config["information-search"].get("temperature", 0.7)
    max_tokens = config["information-search"].get("max_tokens", 4096)
    vector_db_id = config["information-search"].get("vector_db_id", "latest")

    q = queue.Queue()

    print(f"Searching in collection {vector_db_id}")
    print(f"Existing collections: {llama_client.vector_stores.list()}")
    print(f"query: {request.prompt}")

    search_results = llama_client.vector_stores.search(
        vector_store_id=vector_db_id,
        query=request.prompt,
        max_num_results=5,
        search_mode="vector"
    )
    retrieved_chunks = []
    for i, result in enumerate(search_results.data):
        chunk_content = result.content[0].text if hasattr(result, 'content') else str(result)
        retrieved_chunks.append(chunk_content)

    prompt_context = "\n\n".join(retrieved_chunks)

    enhaned_prompt = f"""Please answer the given query using the document intelligence context below.

    CONTEXT (Processed with Docling Document Intelligence):
    {prompt_context}

    QUERY:
    {request.prompt}

    Note: The context includes intelligently processed content with preserved tables, formulas, figures, and document structure."""

    def worker():
        print(f"sending requestion to model {config['summarize']['model']}")
        try:
            response = llama_client.chat.completions.create(
                model=config["summarize"]["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": enhaned_prompt},
                ],
                max_tokens=max_tokens, 
                temperature=temperature,
                stream=True,
            )
            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
                    q.put(chunk)
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)

    threading.Thread(target=worker).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.post("/student-assistant")
async def student_assistant_chat(request: PromptRequest):
    if not FEATURE_FLAGS.get("student-assistant", False):
        raise HTTPException(status_code=404, detail="Student assistant feature is not enabled")

    if not agent:
        raise HTTPException(status_code=500, detail="Student assistant not initialized")

    q = queue.Queue()

    def worker():
        try:
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            thread_id = str(int(random.random() * 1000000))
            config_params = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [HumanMessage(content=request.prompt)]}

            seen_messages = 0
            seen_tool_calls = set()  # Track which tool calls we've already sent

            for state in agent.stream(inputs, config_params, stream_mode="values"):
                messages = state.get("messages", [])
                new_messages = messages[seen_messages:]
                seen_messages = len(messages)

                for msg in new_messages:
                    msg_type = getattr(msg, "type", None)

                    if msg_type == "ai":
                        # Check for tool calls
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                # Create unique identifier for this tool call
                                tool_call_id = f"{tc.get('name')}:{json.dumps(tc.get('args'), sort_keys=True)}"
                                if tool_call_id not in seen_tool_calls:
                                    seen_tool_calls.add(tool_call_id)
                                    tool_call_data = {
                                        "type": "tool_call",
                                        "name": tc.get("name"),
                                        "args": tc.get("args")
                                    }
                                    chunk = f"data: {json.dumps(tool_call_data)}\n\n"
                                    q.put(chunk)
                        else:
                            # Check for MCP tool outputs
                            tool_outputs = msg.additional_kwargs.get("tool_outputs", [])
                            for t in tool_outputs:
                                if t.get("type") == "mcp_call":
                                    mcp_data = {
                                        "type": "mcp_call",
                                        "name": t.get("name"),
                                        "server_label": t.get("server_label"),
                                        "arguments": t.get("arguments"),
                                        "output": t.get("output", ""),
                                        "error": t.get("error", "")
                                    }
                                    chunk = f"data: {json.dumps(mcp_data)}\n\n"
                                    q.put(chunk)

                    elif msg_type == "tool":
                        # Tool result
                        tool_result_data = {
                            "type": "tool_result",
                            "name": msg.name,
                            "content": str(msg.content)  # Show more content
                        }
                        chunk = f"data: {json.dumps(tool_result_data)}\n\n"
                        q.put(chunk)

            # After streaming all intermediate steps, send the final answer
            if messages:
                for msg in reversed(messages):
                    if getattr(msg, "type", None) == "ai":
                        content = msg.content
                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                            content = "".join(text_parts)

                        if content:
                            # Send final answer marker
                            chunk = f"data: {json.dumps({'type': 'final_answer'})}\n\n"
                            q.put(chunk)
                            # Stream the final answer character by character
                            for char in content:
                                chunk = f"data: {json.dumps({'delta': char})}\n\n"
                                q.put(chunk)
                        break
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)

    threading.Thread(target=worker).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)