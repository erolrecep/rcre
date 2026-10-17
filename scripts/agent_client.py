#!/usr/bin/env python3
"""
RCRE Multi-Agent Client Bridge
Connects to local LLM inference engines (Local Agent 1: Qwen3.8-27B on Apple Silicon oMLX,
Local Agent 2: Nemotron-3-Nano on dlbox LM Studio) to assist in code generation,
peer review, test verification, and documentation.
"""

import sys
import json
import urllib.request
import urllib.error

AGENTS_CONFIG = {
    "agent1": {
        "name": "Qwen3.8-27B-8bit (oMLX Local M5 Max)",
        "url": "http://127.0.0.1:8008/v1/chat/completions",
        "model": "Qwen3.8-27B-8bit",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer adana"
        }
    },
    "agent2": {
        "name": "Nemotron-3-Nano-4B (LM Studio dlbox)",
        "url": "http://100.97.208.11:1234/v1/chat/completions",
        "model": "nvidia/nemotron-3-nano-4b",
        "headers": {
            "Content-Type": "application/json"
        }
    }
}

def query_agent(agent_key: str, system_prompt: str, user_prompt: str, max_tokens: int = 512, timeout: int = 30) -> str:
    if agent_key not in AGENTS_CONFIG:
        raise ValueError(f"Unknown agent: {agent_key}. Choose 'agent1' or 'agent2'")

    cfg = AGENTS_CONFIG[agent_key]
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    req = urllib.request.Request(
        cfg["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers=cfg["headers"]
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if not content and "reasoning_content" in msg:
                content = msg["reasoning_content"]
            return content.strip()
    except Exception as e:
        return f"[Agent Query Error ({cfg['name']})]: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 agent_client.py <agent1|agent2> <prompt>")
        sys.exit(1)

    target_agent = sys.argv[1]
    prompt = " ".join(sys.argv[2:])
    print(f"📡 Querying {AGENTS_CONFIG.get(target_agent, {}).get('name', target_agent)}...")
    result = query_agent(target_agent, "You are an expert C23 systems engineer for RCRE.", prompt)
    print("\n--- Agent Response ---")
    print(result)
