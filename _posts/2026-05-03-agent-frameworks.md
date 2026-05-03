---
layout: post
title: "the philosophy of agent frameworks"
date: 2026-05-03 00:00:00 +0530
categories: [tech]
tokens: "~25k"
description: "every agent framework is just a disagreement about how much to trust the model. here's how to actually think about the landscape."
---

<style>
:root {
  --agent-bg: var(--color-background-secondary, #1a1a2e);
  --agent-border: var(--color-border, #2a2a4a);
  --agent-text: var(--color-text-primary, #e8e8e8);
  --agent-muted: var(--color-text-secondary, #888);
  --agent-teal: #4ecdc4;
  --agent-purple: #a78bfa;
  --agent-blue: #4f9ef8;
  --agent-green: #2ecc71;
  --agent-orange: #f08c4b;
  --agent-red: #ff6b6b;
  --agent-yellow: #ffd93d;
}
.agent-video { margin: 1.8rem 0; text-align: center; }
.agent-video video { max-width: 100%; border-radius: 10px; border: 1px solid var(--agent-border); }
.agent-card { background: var(--agent-bg); border: 1px solid var(--agent-border); border-radius: 10px; padding: 16px 20px; margin: 1.2rem 0; font-size: 14px; color: var(--agent-muted); }
.agent-card strong { color: var(--agent-text); }
.agent-highlight { border-left: 3px solid var(--agent-teal); padding: 10px 14px; margin: 1rem 0; font-size: 13px; color: var(--agent-muted); background: var(--agent-bg); border-radius: 0 5px 5px 0; }
.agent-highlight strong { color: var(--agent-teal); }
.agent-img { margin: 1.8rem auto; display: block; max-width: 680px; width: 100%; border-radius: 10px; }
.diag { margin: 1.8rem 0; text-align: center; }
.diag svg { max-width: 720px; width: 100%; }
.diag-caption { font-size: 12px; color: var(--agent-muted); margin-top: 6px; }
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>

> **TL;DR:** An agent is an LLM in a loop with tools. Every framework disagrees about how much of that loop to hard-code vs. let the model decide. The landscape went from chains (2022) to stateful graphs (2024) to model-driven harnesses (2025). MCP won as the integration standard. Pick your framework based on how much you trust the model and how much you need to debug.

A reading guide to LangChain, LangGraph, n8n, Agno, and the broader agent landscape in 2026.

## table of contents

0. [what "agent" actually means](#what-agent-actually-means-in-this-conversation)
1. [how we got here](#how-we-got-here)
2. [the core mental models](#the-core-mental-models)
3. [the frameworks, one by one](#the-frameworks-one-by-one)
4. [the protocol layer](#the-protocol-layer-underneath-all-of-this)
5. [where things are right now](#where-things-actually-are-right-now)
6. [how to actually choose](#how-to-actually-choose)

---

## what "agent" actually means in this conversation

Before any framework makes sense, you have to settle the word. In this post, an agent is an LLM running in a loop with access to tools, deciding for itself what to do next. The minimum viable agent is the ReAct pattern from 2022: model produces a thought, picks a tool, sees the tool result, picks the next tool, and so on until it decides it's done. That is the whole idea. Everything else, every framework, every architecture diagram, every "multi-agent system," is disagreement about how much of that loop you should hard-code versus how much you should let the model decide.

<div class="agent-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/agents/agents_react_loop_poster.jpg">
  <source src="/assets/images/agents/agents_react_loop.mp4" type="video/mp4">
</video>
</div>

That single axis, how much control the developer keeps versus how much the developer cedes to the model, is the most useful lens for understanding the space. On one end, you have rigid pipelines where the developer writes every step and the LLM is just a smart string transformer at each node. On the other end, you have a single big "do the thing" call where the model is trusted to plan, decompose, retry, and finish on its own. Every framework you've heard of sits somewhere on that axis, and most have moved along it as models got better. When a smarter model can replace the framework, the team behind it has to keep redefining what value they add.

<svg width="100%" viewBox="0 0 680 150" xmlns="http://www.w3.org/2000/svg" role="img" style="max-width: 680px; display: block; margin: 1.5rem auto;">
  <title>Control spectrum of agent architectures</title>
  <desc>Five architectural categories arranged from developer-controlled flow on the left to model-controlled flow on the right.</desc>
  <style>
    .label { fill: #5F5E5A; font-family: system-ui, sans-serif; font-size: 12px; }
    .axis-stroke { stroke: #5F5E5A; }
    .box-fill { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 0.5; }
    .title { fill: #2C2C2A; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 500; }
    .subtitle { fill: #5F5E5A; font-family: system-ui, sans-serif; font-size: 12px; }
    @media (prefers-color-scheme: dark) {
      .label { fill: #B4B2A9; }
      .axis-stroke { stroke: #B4B2A9; }
      .box-fill { fill: #444441; stroke: #D3D1C7; }
      .title { fill: #F1EFE8; }
      .subtitle { fill: #D3D1C7; }
    }
  </style>
  <defs>
    <marker id="arrow-spec" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" class="axis-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="40" y="38" class="label">Developer drives flow</text>
  <text x="640" y="38" class="label" text-anchor="end">Model drives flow</text>
  <line x1="180" y1="34" x2="500" y2="34" class="axis-stroke" stroke-width="1.5" marker-end="url(#arrow-spec)"/>
  <rect x="40" y="70" width="100" height="60" rx="8" class="box-fill"/>
  <text x="90" y="92" class="title" text-anchor="middle" dominant-baseline="central">Chains</text>
  <text x="90" y="112" class="subtitle" text-anchor="middle" dominant-baseline="central">Sequential</text>
  <rect x="165" y="70" width="100" height="60" rx="8" class="box-fill"/>
  <text x="215" y="92" class="title" text-anchor="middle" dominant-baseline="central">Graphs</text>
  <text x="215" y="112" class="subtitle" text-anchor="middle" dominant-baseline="central">With cycles</text>
  <rect x="290" y="70" width="100" height="60" rx="8" class="box-fill"/>
  <text x="340" y="92" class="title" text-anchor="middle" dominant-baseline="central">Teams</text>
  <text x="340" y="112" class="subtitle" text-anchor="middle" dominant-baseline="central">Role-based</text>
  <rect x="415" y="70" width="100" height="60" rx="8" class="box-fill"/>
  <text x="465" y="92" class="title" text-anchor="middle" dominant-baseline="central">Dialogue</text>
  <text x="465" y="112" class="subtitle" text-anchor="middle" dominant-baseline="central">Multi-agent</text>
  <rect x="540" y="70" width="100" height="60" rx="8" class="box-fill"/>
  <text x="590" y="92" class="title" text-anchor="middle" dominant-baseline="central">Harness</text>
  <text x="590" y="112" class="subtitle" text-anchor="middle" dominant-baseline="central">LLM-driven</text>
</svg>

---

## how we got here

The pre-history is short and worth knowing. Before late 2022, "agents" mostly meant reinforcement learning agents in research labs, which is a different field. The current usage of the word starts with the ReAct paper (Yao et al., late 2022) and explodes in early 2023 with AutoGPT and BabyAGI, neither of which were good but both of which planted the meme that you could let GPT-4 loop on itself with tools and have it do real things. Most of those early systems failed the same way: they would get into loops, lose the plot, or burn through tokens chasing a hallucinated subgoal.

LangChain shows up in this period as the first widely used Python library that gave you primitives for chaining LLM calls together, plugging in tools, doing retrieval, and so on. It became the dominant library because it was the easiest way to go from "I have an OpenAI key" to "I have a working RAG system." Senior engineers criticized it for hiding prompts behind layers of abstraction. By mid-2023 many had concluded that calling the API direct was simpler than wrestling with LangChain's wrappers. The team rewrote it in 2025 to be leaner.

The second wave, late 2023 through 2024, was the realization that real agents need state, not just chains. A chain is acyclic: input goes in, output comes out, no loops, no memory between runs. A real agent needs to be able to revisit a step, retry on failure, wait for a human, persist conversation history, and maintain typed state across many tool calls. Chains can't do this cleanly. LangGraph was the LangChain team's answer. CrewAI took a different bet: instead of explicit state graphs, model agents as specialists in a team. AutoGen, from Microsoft Research, took a third bet: model the system as a multi-turn conversation between agents.

The third wave, 2025 into 2026, is the one we're in now. Once Claude 3.5/4 and GPT-4o/5 became reliable enough at tool use that you could give them tools and let them loop, Anthropic, OpenAI, AWS, and LangChain each shipped "agent harness" frameworks that don't try to control the loop at all: Claude Agent SDK, Agents SDK (replacing Swarm), Strands Agents, and deepagents. All four bet that the model is now good enough that you should hand it the keys and focus on what tools and memory it has access to. At the same time, the protocol layer matured: MCP, introduced by Anthropic in November 2024, became the closest thing the field has to a universal standard.

<svg width="100%" viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" role="img" style="max-width: 680px; display: block; margin: 1.5rem auto;">
  <title>Three generations of agent frameworks</title>
  <desc>Evolution from chain-based agents in 2022-23 through stateful agents in 2024 to model-driven harnesses in 2025-26.</desc>
  <style>
    .axis-stroke { stroke: #5F5E5A; }
    .gray-fill { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 0.5; }
    .gray-title { fill: #2C2C2A; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 500; }
    .gray-sub { fill: #5F5E5A; font-family: system-ui, sans-serif; font-size: 12px; }
    .purple-fill { fill: #EEEDFE; stroke: #534AB7; stroke-width: 0.5; }
    .purple-title { fill: #26215C; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 500; }
    .purple-sub { fill: #534AB7; font-family: system-ui, sans-serif; font-size: 12px; }
    .teal-fill { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.5; }
    .teal-title { fill: #04342C; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 500; }
    .teal-sub { fill: #0F6E56; font-family: system-ui, sans-serif; font-size: 12px; }
    .pill { fill: #FFFFFF; stroke: #B4B2A9; stroke-width: 0.5; }
    .pill-text { fill: #5F5E5A; font-family: system-ui, sans-serif; font-size: 12px; }
    @media (prefers-color-scheme: dark) {
      .axis-stroke { stroke: #B4B2A9; }
      .gray-fill { fill: #444441; stroke: #D3D1C7; }
      .gray-title { fill: #F1EFE8; }
      .gray-sub { fill: #D3D1C7; }
      .purple-fill { fill: #3C3489; stroke: #CECBF6; }
      .purple-title { fill: #EEEDFE; }
      .purple-sub { fill: #CECBF6; }
      .teal-fill { fill: #085041; stroke: #9FE1CB; }
      .teal-title { fill: #E1F5EE; }
      .teal-sub { fill: #9FE1CB; }
      .pill { fill: #2C2C2A; stroke: #888780; }
      .pill-text { fill: #D3D1C7; }
    }
  </style>
  <defs>
    <marker id="arrow-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" class="axis-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="50" y="40" width="160" height="56" rx="8" class="gray-fill"/>
  <text x="130" y="62" class="gray-title" text-anchor="middle" dominant-baseline="central">Origins</text>
  <text x="130" y="82" class="gray-sub" text-anchor="middle" dominant-baseline="central">2022 - 2023</text>
  <rect x="50" y="110" width="160" height="30" rx="6" class="pill"/>
  <text x="130" y="125" class="pill-text" text-anchor="middle" dominant-baseline="central">ReAct paper</text>
  <rect x="50" y="148" width="160" height="30" rx="6" class="pill"/>
  <text x="130" y="163" class="pill-text" text-anchor="middle" dominant-baseline="central">AutoGPT, BabyAGI</text>
  <rect x="50" y="186" width="160" height="30" rx="6" class="pill"/>
  <text x="130" y="201" class="pill-text" text-anchor="middle" dominant-baseline="central">LangChain v0</text>
  <line x1="215" y1="68" x2="255" y2="68" class="axis-stroke" stroke-width="1.5" marker-end="url(#arrow-gen)"/>
  <rect x="260" y="40" width="160" height="56" rx="8" class="purple-fill"/>
  <text x="340" y="62" class="purple-title" text-anchor="middle" dominant-baseline="central">Stateful agents</text>
  <text x="340" y="82" class="purple-sub" text-anchor="middle" dominant-baseline="central">2024</text>
  <rect x="260" y="110" width="160" height="30" rx="6" class="pill"/>
  <text x="340" y="125" class="pill-text" text-anchor="middle" dominant-baseline="central">LangGraph</text>
  <rect x="260" y="148" width="160" height="30" rx="6" class="pill"/>
  <text x="340" y="163" class="pill-text" text-anchor="middle" dominant-baseline="central">CrewAI, AutoGen</text>
  <rect x="260" y="186" width="160" height="30" rx="6" class="pill"/>
  <text x="340" y="201" class="pill-text" text-anchor="middle" dominant-baseline="central">n8n adds AI</text>
  <rect x="260" y="224" width="160" height="30" rx="6" class="pill"/>
  <text x="340" y="239" class="pill-text" text-anchor="middle" dominant-baseline="central">MCP launched</text>
  <line x1="425" y1="68" x2="465" y2="68" class="axis-stroke" stroke-width="1.5" marker-end="url(#arrow-gen)"/>
  <rect x="470" y="40" width="160" height="56" rx="8" class="teal-fill"/>
  <text x="550" y="62" class="teal-title" text-anchor="middle" dominant-baseline="central">Agent harnesses</text>
  <text x="550" y="82" class="teal-sub" text-anchor="middle" dominant-baseline="central">2025 - 2026</text>
  <rect x="470" y="110" width="160" height="30" rx="6" class="pill"/>
  <text x="550" y="125" class="pill-text" text-anchor="middle" dominant-baseline="central">OpenAI Agents SDK</text>
  <rect x="470" y="148" width="160" height="30" rx="6" class="pill"/>
  <text x="550" y="163" class="pill-text" text-anchor="middle" dominant-baseline="central">Claude Agent SDK</text>
  <rect x="470" y="186" width="160" height="30" rx="6" class="pill"/>
  <text x="550" y="201" class="pill-text" text-anchor="middle" dominant-baseline="central">Strands, Google ADK</text>
  <rect x="470" y="224" width="160" height="30" rx="6" class="pill"/>
  <text x="550" y="239" class="pill-text" text-anchor="middle" dominant-baseline="central">Agno, deepagents</text>
</svg>

<div class="agent-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/agents/agents_evolution_poster.jpg">
  <source src="/assets/images/agents/agents_evolution.mp4" type="video/mp4">
</video>
</div>

---

## the core mental models

A handful of architectural metaphors cover the entire space.

The first is the **chain or pipeline**. You wire up a directed acyclic graph of LLM calls and other operations, and data flows through. The developer specifies the structure, and the LLM is just the smart bit at each step. This is what early LangChain was. It works for fixed pipelines but doesn't qualify as agentic.

The second is the **stateful graph or state machine**. Same idea as a chain, but you allow cycles, conditional edges, and explicit shared state that flows through the graph and gets updated at each node. Now the LLM can decide, at certain nodes, which edge to take next. The developer still defines the topology, but the LLM steers within it. LangGraph is the example. This is the dominant mental model for production systems where you need to know exactly what your agent did and why.

The third is the **role-based team**. You define agents as specialists with roles, goals, backstories, and tools, and you give them tasks. The framework handles delegation, communication, and result aggregation. You don't think about graphs; you think about a marketing team or a research crew. CrewAI is the example. This makes the common case fast to prototype and the uncommon case painful, because the abstraction is rigid.

The fourth is the **multi-agent conversation**. Multiple agents, each with their own system prompt and tools, interact through a shared dialogue. An orchestrator decides who speaks next. AutoGen (now AG2) is built on this. It's good for problems where you want emergent behavior from agents debating, refining each other's outputs, or specializing through dialogue, and most agentic research happens here.

The fifth is the **agent harness or tool-calling loop**. A harness here means a thin wrapper around the model: it handles the plumbing (streaming, retries, tool dispatch) but makes zero decisions about what the agent does next. There is no graph, no team, no conversation: just one LLM in a loop, with a curated set of tools and a long-context memory, trusted to drive. The framework makes that loop production-grade: streaming, persistence, observability, sub-agents on demand, file-system-as-memory, planning guides. Claude Agent SDK, OpenAI Agents SDK, deepagents, and Strands all live here. This is the bet on capability: as models improve, less scaffolding is needed.

<div class="agent-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/agents/agents_chain_graph_harness_poster.jpg">
  <source src="/assets/images/agents/agents_chain_graph_harness.mp4" type="video/mp4">
</video>
</div>

The sixth is the **visual workflow builder**. You drag nodes onto a canvas, connect them with arrows, and the canvas itself is the program. AI agents are just one type of node alongside HTTP calls, database queries, conditionals, and so on. n8n, Make, Zapier, and similar tools live here. The philosophy is that most real-world automations are mostly deterministic with AI sprinkled in, and a visual interface meets that reality better than code.

These are not mutually exclusive. LangGraph can run inside an n8n node. An OpenAI Agents SDK agent can be a sub-agent in a CrewAI crew.

---

## one task, three frameworks

The mental models above are easier to feel when you see the same task implemented three ways. Take a simple agent that answers questions by searching the web and then synthesizing a response. Here it is in LangGraph, Agno, and the Claude Agent SDK.

**LangGraph: you draw the graph**

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict

class State(TypedDict):
    question: str
    search_results: str
    answer: str

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

def search(state: State) -> State:
    # call your search tool, store results in state
    results = web_search(state["question"])
    return {"search_results": results}

def synthesize(state: State) -> State:
    prompt = f"Answer based on: {state['search_results']}\nQ: {state['question']}"
    answer = llm.invoke(prompt).content
    return {"answer": answer}

def should_search(state: State) -> str:
    return "search" if needs_search(state["question"]) else "synthesize"

graph = StateGraph(State)
graph.add_node("search", search)
graph.add_node("synthesize", synthesize)
graph.set_conditional_entry_point(should_search)
graph.add_edge("search", "synthesize")
graph.add_edge("synthesize", END)
agent = graph.compile()

result = agent.invoke({"question": "What is MCP?"})
```

You defined the topology. You chose the edges. The LLM fills in the blanks at each node, but it never decides *which node to visit next* - that's your conditional function. If something goes wrong, you look at the state at each transition and see exactly where it derailed.

**Agno: you hand it tools**

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-20250514"),
    tools=[DuckDuckGoTools()],
    instructions="Answer questions. Search the web if you need current info.",
    markdown=True,
)

agent.print_response("What is MCP?")
```

Six lines. No graph, no state class, no edges. The model decides whether to search. Agno wraps the tool-calling loop and handles the back-and-forth internally. You trade visibility for speed of development.

**Claude Agent SDK: you trust the model**

```python
import anthropic

client = anthropic.Anthropic()
tools = [web_search_tool]  # MCP-compatible tool definitions

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="Answer questions. Search if needed. Think step by step.",
    tools=tools,
    messages=[{"role": "user", "content": "What is MCP?"}],
)

# The SDK handles the tool-use loop:
# model calls search -> gets results -> synthesizes answer
```

The model does everything. You provide tools and a system prompt. The SDK runs the tool-calling loop until the model decides it's done. This is the harness philosophy: the model is the control flow.

<div class="agent-highlight">
<strong>The same task, three levels of developer control.</strong> LangGraph: you hard-code the flow. Agno: you configure the agent and let it loop. Claude SDK: you hand over the keys. Each is the right choice in different contexts. The question is always how much you trust the model and how much you need to debug.
</div>

---

## the frameworks, one by one

### LangChain

LangChain started as a chain library, became an ecosystem, and is currently rebranding as "the agent engineering platform." The original 2023 LangChain was a giant collection of integrations (every vector store, every LLM, every tool) wrapped in a set of abstractions for chaining LLM calls. It was, in retrospect, too opinionated. The team rewrote it in 2025 to be more streamlined, and the modern langchain package is a much leaner integration layer on top of langgraph's runtime.

The mistake people make about LangChain is treating it as a single thing. Think of it as three layers stacked. At the bottom is langchain-core, which gives you common abstractions for messages, models, tools, and runnables. In the middle is langchain itself, the integration layer with hundreds of model providers, vector stores, and tools. On top sits langgraph for actual agent control flow, langsmith for observability, and langserve for deployment. When people say "LangChain is bloated," they usually mean the integration layer. When people say "LangChain is the most powerful framework," they usually mean the whole stack including LangGraph and LangSmith. Both are true.

The current value proposition is breadth. If you need to talk to 100 different models, 50 different vector stores, and 200 different APIs, nobody else has anything close. The cost is that every abstraction you adopt is one more layer between you and the actual prompt being sent. For a senior engineer, that's often a worse tradeoff than just calling the model API directly and writing your own thin wrapper.

---

### LangGraph

LangGraph is the part of the LangChain world that most production teams care about. The model is simple and worth understanding precisely: your agent is a directed graph; nodes are functions (which can call LLMs, tools, or anything else); edges are control flow (which can be conditional based on state); and a single typed State object flows through the graph and is updated by nodes. Cycles are allowed, so an agent can loop. Checkpointing is built in, so every state transition is persisted, which gives you time-travel debugging, human-in-the-loop pauses, and crash recovery.

LangGraph won as the production default because it makes the control flow explicit. When something goes wrong in production, you can look at the graph, look at the state at each transition, and reason about what the agent did. With a "trust the model" harness, you have to read traces and try to figure out why the model made a given decision, which is harder. LangGraph's bet is that for high-stakes systems, the cost of explicitness is lower than the cost of opacity.

The downside is that explicitness. You write more code than you would in CrewAI or Agno for the same prototype. The learning curve is real. And as models improve, some of the structure you encoded in your graph becomes structure the model could have figured out on its own, meaning you've over-engineered. The LangChain team's answer is deepagents, which is a higher-level abstraction built on top of langgraph's runtime that gives you the "harness" feel without giving up the underlying durability and statefulness. It's LangChain's equivalent to Claude Agent SDK.

If you only learn one framework for production work, LangGraph is the choice today. It is model-agnostic & open-sourced.

---

### n8n

n8n is the odd one in the list, because it is not an agent framework. It is a general-purpose workflow automation platform, like Zapier or Make, that has added AI agent nodes. The philosophy is the inverse of code-first frameworks: most real-world automations are deterministic plumbing with AI in a few key places, and a visual builder is the right abstraction for that reality.

The architecture matters. n8n is a TypeScript application, and its AI agent nodes are built on LangChain.js underneath. So when you drop an "AI Agent" node onto an n8n canvas and wire up a chat model, a memory, and some tools, you are configuring a LangChain agent through a GUI. This is why n8n's own internal AI Workflow Builder, the feature that generates workflows from natural language prompts, is itself a LangGraph multi-agent system under the hood. The visual layer sits on top of code-first frameworks; it doesn't replace them.

The right way to think about n8n is as a hybrid runtime. You get deterministic nodes (HTTP calls, database queries, scheduling, branching, error handling) for the boring parts of any automation, and you get AI agent nodes for the parts that need a model. The model isn't doing everything; it's doing the part that has to be smart, while the deterministic graph around it handles auth, retries, error paths, and integration.

It also means n8n's sweet spot is the workflows where you need both reliability and AI: the right move is often to keep the heavy reasoning inside a LangGraph or Agno agent and call it from n8n as a single node.

---

### Agno

Agno (formerly Phi Data, rebranded in 2024) is a reaction against what its authors saw as overengineering in LangChain and LangGraph. Its pitch is "pure Python, no graphs, no chains, just agents that work and run fast."

The Agent class encapsulates the entire reasoning loop in a single object. You construct it with a model, tools, memory, knowledge sources, and storage; you call it; it runs the loop and returns a result. There is no graph to define, no edges to wire. For multi-agent systems, Agno gives you Teams (with four coordination modes: route, coordinate, collaborate, and a couple of variants) and Workflows for sequenced execution. Each team member can itself be an agent or a sub-team, so you get composition for free.

The technical bet that distinguishes it is performance and statelessness. Agno claims agent instantiation in around 3 microseconds and a tiny memory footprint, achieved by treating agents as lightweight, stateless, session-scoped objects rather than long-lived stateful processes. This is the right design for horizontal scaling: spin up an agent per request, do the work, throw it away. State lives in the storage layer (Postgres, SQLite, Mongo, vector stores) rather than in the agent. This is a different model from LangGraph, where agents are graph runtimes that maintain state internally and rely on checkpointing for durability.

AgentOS, the runtime layer, exposes agents, teams, and workflows as REST endpoints with built-in OpenAPI documentation, session management, streaming, and observability hooks. You write an Agent in Python; AgentOS gives you a deployable service. There's also an Agno-Go port that brings the same design to Go for teams who need real concurrency.

Where Agno makes sense over LangGraph is when you want the developer experience of pure Python, you don't need the visual graph abstraction, you care about cold-start latency and per-agent cost, and the integration ecosystem you need is already covered (it has 100+ integrations and supports MCP). Where LangGraph still wins is when your control flow is genuinely complex and you want it visible, when you need the durability guarantees of explicit checkpointing, or when you're already in the LangChain ecosystem and the migration cost is high.

---

### CrewAI

CrewAI's single big idea is that multi-agent problems map onto teams of human specialists, so just code that metaphor directly. You define agents as specialists with a role ("Senior Researcher") and a set of tools. You define tasks. You assemble them into a Crew. You run it. The framework handles delegation, agent-to-agent context passing, and final aggregation.

This is fast for the common case and frustrating for the uncommon case. If your problem fits the metaphor (research, writing, analysis pipelines, content generation, anything you might describe as "a small team of people each doing their part"), you're in production in an afternoon. If your problem doesn't fit, the framework gets in your way. CrewAI's recent versions have added MCP support.

It's the right pick for fast prototyping of business agents and for teams who want to ship without learning graph theory.

---

### AutoGen and AG2

AutoGen came out of Microsoft Research and was always more research-oriented than production-oriented. The original v0.2 introduced the idea of agents talking to each other in a multi-turn conversation, with agents debating and refining each other's outputs. The v0.4 rewrite, now branded AG2, rearchitected the system to be event-driven, async-first, and with pluggable orchestration strategies. The key abstraction is GroupChat: multiple agents in a shared conversation, with a selector function that decides who speaks next.

AutoGen's bet is that emergent intelligence comes from agent dialogue. Agents specialize through their roles, the conversation history is the shared state, and the selector is the control flow. This is a different bet from CrewAI's task-based delegation: in AutoGen, the conversation is the orchestration. AutoGen Studio gives you a low-code interface on top, but most serious AutoGen users write code.

It remains the framework of choice for research-style multi-agent experiments, code-generation systems where you want a coder agent and a critic agent going back and forth, and anywhere you want maximum flexibility for orchestration patterns. In production it requires more DIY infrastructure than LangGraph or Agno.

---

### OpenAI Agents SDK

OpenAI's Agents SDK launched in March 2025 as a production-grade replacement for their experimental Swarm framework. The key abstraction is the handoff: agents transfer control to each other explicitly, carrying conversation context through the transition. Each agent declares its instructions, model, tools, and the list of agents it can hand off to. The runtime handles the routing.

This is a minimalist framework. Compared to LangGraph's typed-state graphs or CrewAI's role-based crews, the Agents SDK gives you few primitives: agents, handoffs, tools, guardrails. The bet is that good models turn simple primitives into rich behavior, and the framework should stay out of the way. It's tightly integrated with OpenAI's models (though it supports other providers), and it ships with good tracing and observability out of the box.

It's a strong choice if you're already on OpenAI infrastructure. It's not a strong choice if you need cross-provider portability.

---

### Claude Agent SDK and deepagents

Claude Agent SDK (Anthropic, 2025) and LangChain's deepagents are the clearest examples of the agent-harness philosophy. The frame is: don't try to encode the control flow at all. Instead, give the agent a strong system prompt, a curated set of tools, a file system as scratch memory, and a planning structure. Then run the LLM in a loop and let it drive. Sub-agents are spawned on demand by the main agent, not pre-wired by the developer.

This works now because Claude 4 and GPT-5 are far better at long-horizon planning, tool selection, and self-correction than their predecessors. The framework's job shifted from "prop the model into reliability" to "remove obstacles so the model can be reliable on its own." File-system-as-memory, in particular, is a clever trick: instead of stuffing everything into the context window, the agent reads and writes files, which lets it work over arbitrarily long horizons without context pressure.

This is the most "bullish on models" position in the field, and it's the right philosophy for agent tasks where you don't know what control flow is needed in advance: open-ended research, code generation, complex debugging. It's the wrong philosophy when you do know the control flow, because hard-coding it is cheaper, more debuggable, and more reliable.

---

### the rest of the landscape

These frameworks matter but don't need the long-form treatment. Each occupies a clear niche:

| Framework | Backer | Mental model | Sweet spot |
|-----------|--------|-------------|------------|
| **Strands** | AWS | Harness | Enterprise agents inside IAM/Bedrock/CloudWatch fabric |
| **Google ADK** | Google | Hierarchical tree | Gemini-native agents with A2A protocol and multimodal input |
| **Pydantic AI** | Pydantic team | Harness | Type-strict Python codebases with dependency injection |
| **smolagents** | HuggingFace | Code-as-action | Agents that write and execute Python instead of JSON tool calls |
| **LlamaIndex** | LlamaIndex | RAG + agents | Retrieval-heavy problems where getting the right context matters most |
| **Semantic Kernel** | Microsoft | Enterprise harness | .NET ecosystem, Azure-integrated deployments |

Two things worth noting from this table. Smolagents' "agents think in code" approach is significant: instead of producing JSON tool calls, the agent generates Python that runs in a sandbox. This compresses both input tokens (no tool schemas) and intermediate state (results flow through variables, not context). Anthropic has been writing about the same idea under "code execution with MCP." And LlamaIndex's retrieval primitives remain best-in-class if your problem is fundamentally about getting the right information into the model.

---

## the protocol layer underneath all of this

The most important development, more important than any individual framework, is the emergence of protocols that sit beneath the frameworks. The two that matter are MCP and A2A.

<svg width="100%" viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" role="img" style="max-width: 680px; display: block; margin: 1.5rem auto;">
  <title>The agent stack</title>
  <desc>Three-layer stack: frameworks at the top, integration protocols in the middle, foundation models at the bottom.</desc>
  <style>
    .teal-fill { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.5; }
    .teal-title { fill: #04342C; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 500; }
    .teal-sub { fill: #0F6E56; font-family: system-ui, sans-serif; font-size: 12px; }
    .amber-fill { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.5; }
    .amber-title { fill: #412402; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 500; }
    .amber-sub { fill: #854F0B; font-family: system-ui, sans-serif; font-size: 12px; }
    .gray-fill { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 0.5; }
    .gray-title { fill: #2C2C2A; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 500; }
    .gray-sub { fill: #5F5E5A; font-family: system-ui, sans-serif; font-size: 12px; }
    .pill { fill: #FFFFFF; stroke: #B4B2A9; stroke-width: 0.5; }
    .pill-text { fill: #5F5E5A; font-family: system-ui, sans-serif; font-size: 12px; }
    @media (prefers-color-scheme: dark) {
      .teal-fill { fill: #085041; stroke: #9FE1CB; }
      .teal-title { fill: #E1F5EE; }
      .teal-sub { fill: #9FE1CB; }
      .amber-fill { fill: #633806; stroke: #FAC775; }
      .amber-title { fill: #FAEEDA; }
      .amber-sub { fill: #FAC775; }
      .gray-fill { fill: #444441; stroke: #D3D1C7; }
      .gray-title { fill: #F1EFE8; }
      .gray-sub { fill: #D3D1C7; }
      .pill { fill: #2C2C2A; stroke: #888780; }
      .pill-text { fill: #D3D1C7; }
    }
  </style>
  <rect x="40" y="40" width="600" height="60" rx="12" class="teal-fill"/>
  <text x="60" y="64" class="teal-title" dominant-baseline="central">Frameworks</text>
  <text x="60" y="84" class="teal-sub" dominant-baseline="central">Control flow, orchestration</text>
  <rect x="295" y="59" width="80" height="22" rx="6" class="pill"/>
  <text x="335" y="70" class="pill-text" text-anchor="middle" dominant-baseline="central">LangGraph</text>
  <rect x="385" y="59" width="50" height="22" rx="6" class="pill"/>
  <text x="410" y="70" class="pill-text" text-anchor="middle" dominant-baseline="central">Agno</text>
  <rect x="445" y="59" width="40" height="22" rx="6" class="pill"/>
  <text x="465" y="70" class="pill-text" text-anchor="middle" dominant-baseline="central">n8n</text>
  <rect x="495" y="59" width="90" height="22" rx="6" class="pill"/>
  <text x="540" y="70" class="pill-text" text-anchor="middle" dominant-baseline="central">Agent SDKs</text>
  <rect x="40" y="120" width="600" height="60" rx="12" class="amber-fill"/>
  <text x="60" y="144" class="amber-title" dominant-baseline="central">Protocols</text>
  <text x="60" y="164" class="amber-sub" dominant-baseline="central">Tool and agent integration</text>
  <rect x="430" y="139" width="60" height="22" rx="6" class="pill"/>
  <text x="460" y="150" class="pill-text" text-anchor="middle" dominant-baseline="central">MCP</text>
  <rect x="500" y="139" width="60" height="22" rx="6" class="pill"/>
  <text x="530" y="150" class="pill-text" text-anchor="middle" dominant-baseline="central">A2A</text>
  <rect x="40" y="200" width="600" height="60" rx="12" class="gray-fill"/>
  <text x="60" y="224" class="gray-title" dominant-baseline="central">Models</text>
  <text x="60" y="244" class="gray-sub" dominant-baseline="central">Foundation LLMs</text>
  <rect x="280" y="219" width="65" height="22" rx="6" class="pill"/>
  <text x="312" y="230" class="pill-text" text-anchor="middle" dominant-baseline="central">Claude</text>
  <rect x="355" y="219" width="50" height="22" rx="6" class="pill"/>
  <text x="380" y="230" class="pill-text" text-anchor="middle" dominant-baseline="central">GPT</text>
  <rect x="415" y="219" width="65" height="22" rx="6" class="pill"/>
  <text x="447" y="230" class="pill-text" text-anchor="middle" dominant-baseline="central">Gemini</text>
  <rect x="490" y="219" width="100" height="22" rx="6" class="pill"/>
  <text x="540" y="230" class="pill-text" text-anchor="middle" dominant-baseline="central">Open weight</text>
</svg>

<div class="agent-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/agents/agents_mcp_flow_poster.jpg">
  <source src="/assets/images/agents/agents_mcp_flow.mp4" type="video/mp4">
</video>
</div>

MCP, the Model Context Protocol, was introduced by Anthropic in November 2024 to solve what they called the N x M integration problem: every model times every tool times every data source equals a custom integration to write. MCP defines a standard JSON-RPC protocol between an MCP client (your AI application) and an MCP server (a tool or data source), so a tool implemented once works with any MCP-compatible client. The original spec defined three primitives (tools, resources, prompts) and has since expanded to five, with sampling and roots added.

The adoption curve was unusually fast for a standard. By March 2025, OpenAI added MCP support to ChatGPT. By Q3 2025, Microsoft shipped MCP servers for GitHub, Azure, and Microsoft 365. By Q1 2026, Google added MCP to Gemini and Vertex AI. In December 2025, Anthropic donated MCP to the Linux Foundation's new Agentic AI Foundation, jointly governed with Block, OpenAI, AWS, Google, and Microsoft, which removed the last political reason for non-Anthropic players to resist it. The public registry now lists thousands of servers. SDK downloads cross 97 million per month. It is, at this point, the integration standard.

You don't write custom integrations for Slack, GitHub, Postgres, or Google Drive anymore; you point your agent at the appropriate MCP server. Frameworks compete on agent control flow, observability, and developer experience, while the integration layer becomes commodity infrastructure. This is what happened with HTTP for the web and JDBC for databases.

A2A, Agent-to-Agent, is the complement. Where MCP lets an agent talk to tools and data, A2A lets agents talk to other agents across framework boundaries. A LangGraph agent can invoke a CrewAI agent through A2A's standardized task interface. Adoption is still earlier than MCP but trending similarly, and Google's ADK ships with native support.

A second important development at this layer is what Anthropic has been calling "code execution with MCP" and what smolagents has done from the start. The insight is that as the number of available tools grows into the hundreds or thousands, loading every tool definition into the context window becomes prohibitively expensive in tokens. Instead, you expose tools as code on a virtual filesystem, give the agent a search-tools function and a code execution sandbox, and let it pull in only the tools it needs. The agent generates and runs a small script that calls the tools, instead of producing JSON for each tool call. This compresses both the input (tool definitions) and the intermediate state (tool outputs flow through code variables, not through the model's context), and it's how production agents at the high end will increasingly work.

---

## where things actually are right now

Four things define the field right now.

**The harness philosophy is winning.** LangChain itself is now shipping deepagents, an explicitly harness-style API on top of langgraph, while keeping the graph layer underneath for cases where you do need it. As models get better, more orchestration moves into the model and out of the framework. The four most-cited production frameworks are now LangGraph, Claude Agent SDK, OpenAI Agents SDK, and Strands, with CrewAI as the role-based alternative and Agno as the pure-Python performance play.

**MCP won.** If you are designing a new agent system today, assume MCP is your integration layer, not an option you add later.

**Most multi-agent hype was wrong.** Most problems people thought needed multi-agent systems turned out to work better with a single well-equipped agent, good tools, and clear instructions. Multi-agent makes sense for independent specializations (a planner and an executor, or a generator and a critic) and for parallel exploration. LangChain's own published guidance: 80% of real applications work better as a single agent.

<svg width="100%" viewBox="0 0 780 400" xmlns="http://www.w3.org/2000/svg" role="img" style="max-width: 780px; display: block; margin: 1.5rem auto;">
  <title>Single agent vs multi-agent: the 80/20 rule</title>
  <style>
    .sa-title { font-family: system-ui, sans-serif; font-size: 16px; font-weight: 600; }
    .sa-sub { font-family: system-ui, sans-serif; font-size: 12px; }
    .sa-label { font-family: system-ui, sans-serif; font-size: 11px; }
    .sa-check { font-family: system-ui, sans-serif; font-size: 12px; }
    .sa-divider-text { font-family: system-ui, sans-serif; font-size: 14px; font-weight: 600; }
    .sa-fill { fill: #F1EFE8; }
    .sa-stroke { stroke: #5F5E5A; }
    .sa-text { fill: #2C2C2A; }
    .sa-muted { fill: #5F5E5A; }
    @media (prefers-color-scheme: dark) {
      .sa-fill { fill: #444441; }
      .sa-stroke { stroke: #D3D1C7; }
      .sa-text { fill: #F1EFE8; }
      .sa-muted { fill: #B4B2A9; }
    }
  </style>
  <!-- Left panel: Single Agent -->
  <text x="195" y="30" text-anchor="middle" class="sa-title sa-text">Single Agent</text>
  <text x="195" y="48" text-anchor="middle" class="sa-sub sa-muted">(80% of real problems)</text>
  <!-- Center agent circle -->
  <circle cx="195" cy="170" r="45" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="2"/>
  <text x="195" y="175" text-anchor="middle" class="sa-title" fill="#4ecdc4">Agent</text>
  <!-- Tool boxes around the circle -->
  <rect x="130" y="70" width="60" height="28" rx="6" class="sa-fill sa-stroke" stroke-width="0.5"/>
  <text x="160" y="88" text-anchor="middle" class="sa-label sa-text">Search</text>
  <line x1="160" y1="98" x2="175" y2="130" class="sa-stroke" stroke-width="1"/>
  <rect x="225" y="70" width="60" height="28" rx="6" class="sa-fill sa-stroke" stroke-width="0.5"/>
  <text x="255" y="88" text-anchor="middle" class="sa-label sa-text">Code</text>
  <line x1="255" y1="98" x2="220" y2="135" class="sa-stroke" stroke-width="1"/>
  <rect x="68" y="150" width="60" height="28" rx="6" class="sa-fill sa-stroke" stroke-width="0.5"/>
  <text x="98" y="168" text-anchor="middle" class="sa-label sa-text">Files</text>
  <line x1="128" y1="164" x2="150" y2="168" class="sa-stroke" stroke-width="1"/>
  <rect x="265" y="150" width="60" height="28" rx="6" class="sa-fill sa-stroke" stroke-width="0.5"/>
  <text x="295" y="168" text-anchor="middle" class="sa-label sa-text">API</text>
  <line x1="265" y1="164" x2="240" y2="168" class="sa-stroke" stroke-width="1"/>
  <rect x="100" y="230" width="65" height="28" rx="6" class="sa-fill sa-stroke" stroke-width="0.5"/>
  <text x="132" y="248" text-anchor="middle" class="sa-label sa-text">Browser</text>
  <line x1="145" y1="230" x2="175" y2="210" class="sa-stroke" stroke-width="1"/>
  <rect x="225" y="230" width="60" height="28" rx="6" class="sa-fill sa-stroke" stroke-width="0.5"/>
  <text x="255" y="248" text-anchor="middle" class="sa-label sa-text">DB</text>
  <line x1="240" y1="230" x2="215" y2="210" class="sa-stroke" stroke-width="1"/>
  <!-- Checkmarks -->
  <text x="120" y="300" class="sa-check" fill="#2ecc71">&#10003; Simpler debugging</text>
  <text x="120" y="320" class="sa-check" fill="#2ecc71">&#10003; Lower latency</text>
  <text x="120" y="340" class="sa-check" fill="#2ecc71">&#10003; Cheaper</text>
  <!-- Divider -->
  <line x1="390" y1="30" x2="390" y2="360" stroke-dasharray="6,4" class="sa-stroke" stroke-width="1.5"/>
  <text x="390" y="385" text-anchor="middle" class="sa-divider-text sa-muted">The 80/20 Rule</text>
  <!-- Right panel: Multi-Agent -->
  <text x="585" y="30" text-anchor="middle" class="sa-title sa-text">Multi-Agent</text>
  <text x="585" y="48" text-anchor="middle" class="sa-sub sa-muted">(when you actually need it)</text>
  <!-- Three agent circles in a triangle -->
  <circle cx="585" cy="110" r="35" fill="#a78bfa" fill-opacity="0.2" stroke="#a78bfa" stroke-width="2"/>
  <text x="585" y="115" text-anchor="middle" class="sa-label" fill="#a78bfa" font-weight="600">Planner</text>
  <circle cx="510" cy="230" r="35" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="2"/>
  <text x="510" y="235" text-anchor="middle" class="sa-label" fill="#4ecdc4" font-weight="600">Executor</text>
  <circle cx="660" cy="230" r="35" fill="#ff6b6b" fill-opacity="0.2" stroke="#ff6b6b" stroke-width="2"/>
  <text x="660" y="235" text-anchor="middle" class="sa-label" fill="#ff6b6b" font-weight="600">Critic</text>
  <!-- Arrows between agents -->
  <defs>
    <marker id="ah-sa" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" class="sa-stroke" stroke-width="1.5" stroke-linecap="round"/>
    </marker>
  </defs>
  <line x1="560" y1="140" x2="530" y2="198" class="sa-stroke" stroke-width="1.5" marker-end="url(#ah-sa)"/>
  <line x1="545" y1="235" x2="625" y2="235" class="sa-stroke" stroke-width="1.5" marker-end="url(#ah-sa)"/>
  <line x1="640" y1="198" x2="610" y2="140" class="sa-stroke" stroke-width="1.5" marker-end="url(#ah-sa)"/>
  <!-- Use cases -->
  <text x="490" y="300" class="sa-check sa-muted">Parallel exploration</text>
  <text x="490" y="320" class="sa-check sa-muted">Generator + critic pairs</text>
  <text x="490" y="340" class="sa-check sa-muted">Different system prompts</text>
</svg>

**"Agent" is fragmenting into useful subcategories.** Coding agents (Claude Code, Cursor, OpenHands, Aider), browser agents (Claude in Chrome, browser-use), workflow agents (n8n, Make), research agents (deepagents, Claude's research mode), customer-facing agents (support bots, phone agents). The general-purpose framework matters less when the agent is specialized. And across all categories, observability (LangSmith, Langfuse, Arize) and governance (guardrails, audit logs, human-in-the-loop) have become their own markets.

---

## how to actually choose

You usually don't pick one. Real systems combine multiple frameworks: a LangGraph control flow calling MCP tools, embedded inside an n8n workflow for the integration plumbing, with observability through LangSmith or OpenTelemetry. The frameworks are not competing for the same slot in your stack; they're competing for different slots.

That said, if forced to pick a single axis: pick LangGraph if you want maximum control and your problem has a real, knowable control flow structure; pick Agno if you want pure-Python ergonomics, fast iteration, and stateless scaling; pick Claude Agent SDK or deepagents if your problem is open-ended and you want to bet on the model; pick CrewAI if you have a team-of-specialists mental model and want to ship a prototype this week; pick n8n if your problem is mostly deterministic plumbing with AI in a few spots and a visual builder helps you reason about it; pick OpenAI Agents SDK if you're already deep in OpenAI's stack and want the path of least resistance. Skip AutoGen unless you're doing research or specifically need multi-agent dialogue.

<svg width="100%" viewBox="0 0 820 520" xmlns="http://www.w3.org/2000/svg" role="img" style="max-width: 820px; display: block; margin: 1.5rem auto;">
  <title>Which agent framework should you pick?</title>
  <style>
    .fc-text { font-family: system-ui, sans-serif; font-size: 12px; }
    .fc-q { font-family: system-ui, sans-serif; font-size: 11px; }
    .fc-fw { font-family: ui-monospace, monospace; font-size: 12px; font-weight: 600; }
    .fc-edge { font-family: system-ui, sans-serif; font-size: 10px; }
    .fc-fill { fill: #F1EFE8; }
    .fc-stroke { stroke: #5F5E5A; }
    .fc-text-color { fill: #2C2C2A; }
    .fc-muted { fill: #5F5E5A; }
    .fc-yes { fill: #2ecc71; }
    .fc-no { fill: #ff6b6b; }
    @media (prefers-color-scheme: dark) {
      .fc-fill { fill: #444441; }
      .fc-stroke { stroke: #D3D1C7; }
      .fc-text-color { fill: #F1EFE8; }
      .fc-muted { fill: #B4B2A9; }
    }
  </style>
  <defs>
    <marker id="ah-fc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" class="fc-stroke" stroke-width="1.5" stroke-linecap="round"/>
    </marker>
  </defs>
  <!-- Start node -->
  <rect x="300" y="10" width="220" height="36" rx="18" fill="#4f9ef8" fill-opacity="0.2" stroke="#4f9ef8" stroke-width="1.5"/>
  <text x="410" y="33" text-anchor="middle" class="fc-text" fill="#4f9ef8" font-weight="600">What's your agent problem?</text>
  <line x1="410" y1="46" x2="410" y2="75" class="fc-stroke" stroke-width="1.5" marker-end="url(#ah-fc)"/>
  <!-- Q1: Know control flow? -->
  <rect x="320" y="75" width="180" height="45" rx="6" class="fc-fill fc-stroke" stroke-width="0.5"/>
  <text x="410" y="95" text-anchor="middle" class="fc-q fc-text-color">Know the control flow</text>
  <text x="410" y="112" text-anchor="middle" class="fc-q fc-text-color">in advance?</text>
  <!-- YES right -->
  <line x1="500" y1="97" x2="560" y2="97" class="fc-stroke" stroke-width="1.5" marker-end="url(#ah-fc)"/>
  <text x="530" y="91" text-anchor="middle" class="fc-edge fc-yes">YES</text>
  <!-- NO down -->
  <line x1="410" y1="120" x2="410" y2="175" class="fc-stroke" stroke-width="1.5" marker-end="url(#ah-fc)"/>
  <text x="420" y="150" class="fc-edge fc-no">NO</text>
  <!-- Q2: Mostly deterministic? -->
  <rect x="560" y="75" width="180" height="45" rx="6" class="fc-fill fc-stroke" stroke-width="0.5"/>
  <text x="650" y="95" text-anchor="middle" class="fc-q fc-text-color">Mostly deterministic</text>
  <text x="650" y="112" text-anchor="middle" class="fc-q fc-text-color">plumbing + some AI?</text>
  <!-- YES -> n8n -->
  <line x1="740" y1="97" x2="780" y2="97" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="760" y="91" class="fc-edge fc-yes">Y</text>
  <rect x="755" y="82" width="55" height="30" rx="6" fill="#f08c4b" fill-opacity="0.2" stroke="#f08c4b" stroke-width="1"/>
  <text x="782" y="101" text-anchor="middle" class="fc-fw" fill="#f08c4b">n8n</text>
  <!-- NO down -> Q3 -->
  <line x1="650" y1="120" x2="650" y2="155" class="fc-stroke" stroke-width="1.5" marker-end="url(#ah-fc)"/>
  <text x="660" y="140" class="fc-edge fc-no">N</text>
  <!-- Q3: Need checkpointing? -->
  <rect x="560" y="155" width="180" height="45" rx="6" class="fc-fill fc-stroke" stroke-width="0.5"/>
  <text x="650" y="175" text-anchor="middle" class="fc-q fc-text-color">Need explicit state</text>
  <text x="650" y="192" text-anchor="middle" class="fc-q fc-text-color">+ checkpointing?</text>
  <!-- YES -> LangGraph -->
  <line x1="740" y1="177" x2="770" y2="177" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="755" y="171" class="fc-edge fc-yes">Y</text>
  <rect x="745" y="162" width="75" height="30" rx="6" fill="#a78bfa" fill-opacity="0.2" stroke="#a78bfa" stroke-width="1"/>
  <text x="782" y="181" text-anchor="middle" class="fc-fw" fill="#a78bfa">LangGraph</text>
  <!-- NO -> Agno -->
  <line x1="650" y1="200" x2="650" y2="225" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="660" y="215" class="fc-edge fc-no">N</text>
  <rect x="615" y="225" width="70" height="30" rx="6" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="1"/>
  <text x="650" y="244" text-anchor="middle" class="fc-fw" fill="#4ecdc4">Agno</text>
  <!-- Q4: Open-ended? (NO path from Q1) -->
  <rect x="320" y="175" width="180" height="45" rx="6" class="fc-fill fc-stroke" stroke-width="0.5"/>
  <text x="410" y="195" text-anchor="middle" class="fc-q fc-text-color">Open-ended task?</text>
  <text x="410" y="212" text-anchor="middle" class="fc-q fc-text-color">(research, coding, debug)</text>
  <!-- YES left -->
  <line x1="320" y1="197" x2="260" y2="197" class="fc-stroke" stroke-width="1.5" marker-end="url(#ah-fc)"/>
  <text x="290" y="191" text-anchor="middle" class="fc-edge fc-yes">YES</text>
  <!-- NO down -->
  <line x1="410" y1="220" x2="410" y2="310" class="fc-stroke" stroke-width="1.5" marker-end="url(#ah-fc)"/>
  <text x="420" y="268" class="fc-edge fc-no">NO</text>
  <!-- Q5: Model provider? -->
  <rect x="80" y="175" width="180" height="45" rx="6" class="fc-fill fc-stroke" stroke-width="0.5"/>
  <text x="170" y="195" text-anchor="middle" class="fc-q fc-text-color">Primary model</text>
  <text x="170" y="212" text-anchor="middle" class="fc-q fc-text-color">provider?</text>
  <!-- Three branches down -->
  <line x1="120" y1="220" x2="70" y2="270" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="80" y="243" class="fc-edge fc-text-color">Anthropic</text>
  <line x1="170" y1="220" x2="170" y2="270" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="180" y="250" class="fc-edge fc-text-color">OpenAI</text>
  <line x1="220" y1="220" x2="270" y2="270" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="260" y="243" class="fc-edge fc-text-color">Multi</text>
  <!-- Terminal: Claude SDK -->
  <rect x="10" y="270" width="100" height="30" rx="6" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="1"/>
  <text x="60" y="289" text-anchor="middle" class="fc-fw" fill="#4ecdc4">Claude SDK</text>
  <!-- Terminal: OpenAI SDK -->
  <rect x="120" y="270" width="100" height="30" rx="6" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="1"/>
  <text x="170" y="289" text-anchor="middle" class="fc-fw" fill="#4ecdc4">OpenAI SDK</text>
  <!-- Terminal: Strands -->
  <rect x="230" y="270" width="100" height="30" rx="6" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="1"/>
  <text x="280" y="289" text-anchor="middle" class="fc-fw" fill="#4ecdc4">Strands</text>
  <!-- Q6: Team metaphor? -->
  <rect x="320" y="310" width="180" height="45" rx="6" class="fc-fill fc-stroke" stroke-width="0.5"/>
  <text x="410" y="330" text-anchor="middle" class="fc-q fc-text-color">Team-of-specialists</text>
  <text x="410" y="347" text-anchor="middle" class="fc-q fc-text-color">metaphor fit?</text>
  <!-- YES -> CrewAI -->
  <line x1="500" y1="332" x2="560" y2="332" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="530" y="326" text-anchor="middle" class="fc-edge fc-yes">YES</text>
  <rect x="560" y="317" width="80" height="30" rx="6" fill="#a78bfa" fill-opacity="0.2" stroke="#a78bfa" stroke-width="1"/>
  <text x="600" y="336" text-anchor="middle" class="fc-fw" fill="#a78bfa">CrewAI</text>
  <!-- NO down -> Q7 -->
  <line x1="410" y1="355" x2="410" y2="395" class="fc-stroke" stroke-width="1.5" marker-end="url(#ah-fc)"/>
  <text x="420" y="378" class="fc-edge fc-no">NO</text>
  <!-- Q7: Multi-agent dialogue? -->
  <rect x="320" y="395" width="180" height="45" rx="6" class="fc-fill fc-stroke" stroke-width="0.5"/>
  <text x="410" y="415" text-anchor="middle" class="fc-q fc-text-color">Need multi-agent</text>
  <text x="410" y="432" text-anchor="middle" class="fc-q fc-text-color">dialogue?</text>
  <!-- YES -> AutoGen -->
  <line x1="500" y1="417" x2="560" y2="417" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="530" y="411" text-anchor="middle" class="fc-edge fc-yes">YES</text>
  <rect x="560" y="402" width="90" height="30" rx="6" fill="#a78bfa" fill-opacity="0.2" stroke="#a78bfa" stroke-width="1"/>
  <text x="605" y="421" text-anchor="middle" class="fc-fw" fill="#a78bfa">AutoGen</text>
  <!-- NO -> Start with Agno -->
  <line x1="410" y1="440" x2="410" y2="470" class="fc-stroke" stroke-width="1" marker-end="url(#ah-fc)"/>
  <text x="420" y="458" class="fc-edge fc-no">NO</text>
  <rect x="340" y="470" width="140" height="30" rx="6" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="1"/>
  <text x="410" y="489" text-anchor="middle" class="fc-fw" fill="#4ecdc4">Agno or Claude SDK</text>
</svg>
