# The Philosophy of Agent Frameworks

A reading guide to LangChain, LangGraph, n8n, Agno, and the broader agent landscape as of May 2026.

## What "agent" actually means in this conversation

Before any framework makes sense, you have to settle the word. In this whole conversation, an agent is just an LLM running in a loop with access to tools, deciding for itself what to do next. The minimum viable agent is the ReAct pattern from 2022: model produces a thought, picks a tool, sees the tool result, picks the next tool, and so on until it decides it's done. That is genuinely the whole idea. Everything else, every framework, every architecture diagram, every "multi-agent system," is just disagreement about how much of that loop you should hard-code versus how much you should let the model decide.

That single axis, how much control the developer keeps versus how much the developer cedes to the model, is the single most useful lens for understanding the landscape. On one end, you have rigid pipelines where the developer writes every step and the LLM is just a smart string transformer at each node. On the other end, you have a single big "do the thing" call where the model is trusted to plan, decompose, retry, and finish on its own. Every framework you've heard of sits somewhere on that axis, and most have moved along it as models got better. When a framework's main competitor is a smarter model, the framework has to keep redefining what value it adds.

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

The reason this is interesting and not just academic is that the right point on the axis depends on three things: how reliable the model is at the task, how much the cost of failure matters, and how much you need to debug, audit, or intervene. A research demo can lean on the model. A production system processing financial transactions cannot. Most agent framework design fights are really fights about where on this control axis the default should sit.

## How we got here

The pre-history is short and worth knowing. Before late 2022, "agents" mostly meant reinforcement learning agents in research labs, which is a completely different field. The current usage of the word starts with the ReAct paper (Yao et al., late 2022) and explodes in early 2023 with AutoGPT and BabyAGI, neither of which were good but both of which planted the meme that you could let GPT-4 loop on itself with tools and have it do real things. Most of those early systems were unreliable in the same characteristic way: they would get into loops, lose the plot, or burn through tokens chasing a hallucinated subgoal.

LangChain shows up in this period as the first widely used Python library that gave you primitives for chaining LLM calls together, plugging in tools, doing retrieval, and so on. It became enormously popular because it was the easiest way to go from "I have an OpenAI key" to "I have a working RAG system." It was also, fairly or not, criticized for being too opinionated, too abstract, and for hiding the actual prompts behind layers of abstraction. By mid-2023 a lot of senior engineers had concluded that just calling the API directly was simpler than wrestling with LangChain's abstractions. LangChain has since rewritten itself in 2025 to be much leaner, but the reputation lingers.

The second wave, late 2023 through 2024, was the realization that real agents need state, not just chains. A chain is acyclic: input goes in, output comes out, no loops, no memory between runs. A real agent needs to be able to revisit a step, retry on failure, wait for a human, persist conversation history, and maintain typed state across many tool calls. Chains can't do this cleanly. LangGraph was the LangChain team's answer. CrewAI took a different bet: instead of explicit state graphs, model agents as specialists in a team. AutoGen, from Microsoft Research, took a third bet: model the system as a multi-turn conversation between agents.

The third wave, 2025 into 2026, is the one we're in now. Once Claude 3.5/4 and GPT-4o/5 became reliable enough at tool use that you could just give them tools and let them loop, a class of "agent harness" frameworks emerged that don't try to control the loop at all: Claude Agent SDK from Anthropic, OpenAI's Agents SDK (which replaced their experimental Swarm), AWS's Strands Agents, and LangChain's own deepagents. These all bet that the model is now good enough that you should hand it the keys and focus on what tools and memory it has access to, not on hard-coding the control flow. At the same time, the protocol layer matured: MCP, introduced by Anthropic in November 2024, became the closest thing the field has to a universal standard, and was donated to the Linux Foundation in December 2025 to neutralize it as a standards body asset rather than a vendor's project.

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

That arc, from chains to graphs to harnesses, is the single most important thing to internalize. It's not random fashion. Each generation made sense given the model capabilities of its moment.

## The core mental models

There are really only a handful of architectural metaphors at play across the entire landscape, and once you can name them, every framework slots into one or two.

The first is the chain or pipeline. You wire up a directed acyclic graph of LLM calls and other operations, and data flows through. The developer specifies the structure, and the LLM is just the smart bit at each step. This is what early LangChain was. It's still useful, but it isn't really agentic, because the LLM doesn't decide the control flow.

The second is the stateful graph or state machine. Same idea as a chain, but you allow cycles, conditional edges, and explicit shared state that flows through the graph and gets updated at each node. Now the LLM can decide, at certain nodes, which edge to take next. The developer still defines the topology, but the LLM steers within it. LangGraph is the canonical example. This is the dominant mental model for production systems where you need to know exactly what your agent did and why.

The third is the role-based team. You define agents as specialists with roles, goals, backstories, and tools, and you give them tasks. The framework handles delegation, communication, and result aggregation. You don't think about graphs; you think about a marketing team or a research crew. CrewAI is the canonical example. This makes the common case extremely fast to prototype and the uncommon case painful, because the abstraction is rigid.

The fourth is the multi-agent conversation. Multiple agents, each with their own system prompt and tools, interact through a shared dialogue. A selector or orchestrator decides who speaks next. AutoGen (now AG2) is built on this. It's good for problems where you want emergent behavior from agents debating, refining each other's outputs, or specializing through dialogue, and it's where a lot of agentic research lives.

The fifth is the agent harness or tool-calling loop. There is no graph, no team, no conversation: just one LLM in a loop, with a curated set of tools and a long-context memory, trusted to drive. The framework's job is to make that loop production-grade: streaming, persistence, observability, sub-agents on demand, file-system-as-memory, planning scaffolds. Claude Agent SDK, OpenAI Agents SDK, deepagents, and Strands all live here. This is the bet on capability: as models improve, less scaffolding is needed.

The sixth is the visual workflow builder. You drag nodes onto a canvas, connect them with arrows, and the canvas itself is the program. AI agents are just one type of node alongside HTTP calls, database queries, conditionals, and so on. n8n, Make, Zapier, and similar tools live here. The philosophy is that most real-world automations are mostly deterministic with AI sprinkled in, and a visual interface meets that reality better than code.

These are not mutually exclusive. LangGraph can run inside an n8n node. An OpenAI Agents SDK agent can be a sub-agent in a CrewAI crew. The metaphors compose in practice, but each framework has a primary one that shapes its API and its sweet spot.

## The frameworks, one by one

### LangChain

LangChain started as a chain library, became an ecosystem, and is currently rebranding as "the agent engineering platform." The original 2023 LangChain was a giant collection of integrations (every vector store, every LLM, every tool) wrapped in a set of abstractions for chaining LLM calls. It was, in retrospect, too opinionated. The team rewrote it in 2025 to be more streamlined, and the modern langchain package is a much leaner integration layer on top of langgraph's runtime.

The philosophical mistake people make about LangChain is to treat it as a single thing. It's better understood as three layers stacked. At the bottom is langchain-core, which gives you common abstractions for messages, models, tools, and runnables. In the middle is langchain itself, the integration layer with hundreds of model providers, vector stores, and tools. On top sits langgraph for actual agent control flow, langsmith for observability, and langserve for deployment. When people say "LangChain is bloated," they usually mean the integration layer. When people say "LangChain is the most powerful framework," they usually mean the whole stack including LangGraph and LangSmith. Both are true.

The current value proposition is breadth. If you need to talk to 100 different models, 50 different vector stores, and 200 different APIs, nobody else has anything close. The cost is that every abstraction you adopt is one more layer between you and the actual prompt being sent. For a senior engineer, that's often a worse tradeoff than just calling the model API directly and writing your own thin wrapper.

### LangGraph

LangGraph is the part of the LangChain world that most production teams actually care about. The model is simple and worth understanding precisely: your agent is a directed graph; nodes are functions (which can call LLMs, tools, or anything else); edges are control flow (which can be conditional based on state); and a single typed State object flows through the graph and is updated by nodes. Cycles are allowed, so an agent can loop. Checkpointing is built in, so every state transition is persisted, which gives you time-travel debugging, human-in-the-loop pauses, and crash recovery essentially for free.

The reason LangGraph won as the production default is exactly because it makes the control flow explicit. When something goes wrong in production, you can look at the graph, look at the state at each transition, and reason about what the agent did. With a "trust the model" harness, you have to read traces and try to figure out why the model made a given decision, which is harder. LangGraph's bet is that for high-stakes systems, the cost of explicitness is lower than the cost of opacity.

The downside is exactly that explicitness. You write more code than you would in CrewAI or Agno for the same prototype. The learning curve is real. And as models improve, some of the structure you carefully encoded in your graph becomes structure the model could have figured out on its own, meaning you've over-engineered. LangGraph's recent answer is deepagents, which is a higher-level abstraction built on top of langgraph's runtime that gives you the "harness" feel without giving up the underlying durability and statefulness. It's roughly LangChain's response to Claude Agent SDK.

If you only learn one framework deeply for production work, LangGraph is the defensible choice today. It is model-agnostic, MIT-licensed, has the largest production deployment surface area, and the runtime guarantees (checkpointing, streaming, durability) are genuinely valuable.

### n8n

n8n is the odd one in your list, because it is fundamentally not an agent framework. It is a general-purpose workflow automation platform, like Zapier or Make, that has added AI agent nodes. The philosophy is the inverse of code-first frameworks: most real-world automations are deterministic plumbing with AI in a few key places, and a visual builder is the right abstraction for that reality.

The architecture matters. n8n is a TypeScript application, and its AI agent nodes are built on LangChain.js underneath. So when you drop an "AI Agent" node onto an n8n canvas and wire up a chat model, a memory, and some tools, you are essentially configuring a LangChain agent through a GUI. This is why n8n's own internal AI Workflow Builder, the feature that generates workflows from natural language prompts, is itself a LangGraph multi-agent system under the hood. The visual layer sits on top of code-first frameworks; it doesn't replace them.

The right way to think about n8n philosophically is as a hybrid runtime. You get deterministic nodes (HTTP calls, database queries, scheduling, branching, error handling) for the boring parts of any automation, and you get AI agent nodes for the parts that genuinely need a model. The model isn't doing everything; it's doing the part that has to be smart, while the deterministic graph around it handles auth, retries, error paths, and integration. This is closer to how production systems actually look than the pure-agent fantasy.

It also means n8n's sweet spot is exactly the workflows where you need both reliability and AI: customer support routing, document classification with human review, lead enrichment, content moderation pipelines. Where it fights against itself is anything that needs sophisticated multi-step agent reasoning, because expressing that as a visual graph gets unwieldy fast. Senior engineers tend to find the visual interface limiting for complex logic; the right move is often to keep the heavy reasoning inside a LangGraph or Agno agent and call it from n8n as a single node.

### Agno

Agno (formerly Phi Data, rebranded in 2024) is the most recent serious entrant in the Python agent space, and it represents a deliberate philosophical reaction against what its authors saw as overengineering in LangChain and LangGraph. Its pitch is "pure Python, no graphs, no chains, just agents that work and run fast."

The architecture is genuinely interesting. The Agent class encapsulates the entire reasoning loop in a single object. You construct it with a model, tools, memory, knowledge sources, and storage; you call it; it runs the loop and returns a result. There is no graph to define, no edges to wire. For multi-agent systems, Agno gives you Teams (with four coordination modes: route, coordinate, collaborate, and a couple of variants) and Workflows for sequenced execution. Each team member can itself be an agent or a sub-team, so you get composition for free.

The technical bet that distinguishes it is performance and statelessness. Agno claims agent instantiation in around 3 microseconds and a tiny memory footprint, achieved by treating agents as lightweight, stateless, session-scoped objects rather than long-lived stateful processes. This is the right design for horizontal scaling: spin up an agent per request, do the work, throw it away. State lives in the storage layer (Postgres, SQLite, Mongo, vector stores) rather than in the agent. This is a noticeably different model from LangGraph, where agents are graph runtimes that maintain state internally and rely on checkpointing for durability.

The runtime layer, AgentOS, deserves its own mention because it shows how serious the production focus is: it's a FastAPI application that exposes agents, teams, and workflows as REST endpoints with built-in OpenAPI documentation, session management, streaming, and observability hooks. You write an Agent in Python; AgentOS gives you a deployable service. There's also an Agno-Go port that brings the same design to Go for teams who need real concurrency.

Where Agno makes sense over LangGraph is when you want the developer experience of pure Python, you don't need the visual graph abstraction, you care about cold-start latency and per-agent cost, and the integration ecosystem you need is already covered (it has 100+ integrations and supports MCP). Where LangGraph still wins is when your control flow is genuinely complex and you want it visible, when you need the durability guarantees of explicit checkpointing, or when you're already in the LangChain ecosystem and the migration cost is high.

### CrewAI

CrewAI's single big idea is that multi-agent problems map naturally onto teams of human specialists, so just code that metaphor directly. You define agents as specialists with a role ("Senior Researcher"), a goal, a backstory (which is a real prompt-engineering trick, not a gimmick), and a set of tools. You define tasks. You assemble them into a Crew. You run it. The framework handles delegation, agent-to-agent context passing, and final aggregation.

This is brilliant for the common case and frustrating for the uncommon case. If your problem fits the metaphor (research, writing, analysis pipelines, content generation, anything you might describe as "a small team of people each doing their part"), you're in production in an afternoon. If your problem doesn't fit, the framework gets in your way. CrewAI's recent versions have added MCP support, A2A protocol support, and Flows for more programmatic control, which addresses some of the rigidity, but the core metaphor is still role-based.

In April 2026 it sits at around 45,000 GitHub stars and reportedly powers a large number of daily agent executions in production. It's the right pick for fast prototyping of business agents and for teams who want to ship without learning graph theory.

### AutoGen and AG2

AutoGen came out of Microsoft Research and was always more research-oriented than production-oriented. The original v0.2 introduced the idea of agents talking to each other in a multi-turn conversation, with agents debating and refining each other's outputs. The v0.4 rewrite, now branded AG2, rearchitected the system to be event-driven, async-first, and with pluggable orchestration strategies. The key abstraction is GroupChat: multiple agents in a shared conversation, with a selector function that decides who speaks next.

Philosophically, AutoGen's bet is that emergent intelligence comes from agent dialogue. Agents specialize through their roles, the conversation history is the shared state, and the selector is the control flow. This is a different bet from CrewAI's task-based delegation: in AutoGen, the conversation is the orchestration. AutoGen Studio gives you a low-code interface on top, but most serious AutoGen users write code.

It remains the framework of choice for research-style multi-agent experiments, code-generation systems where you want a coder agent and a critic agent going back and forth, and anywhere you want maximum flexibility for orchestration patterns. In production it requires more DIY infrastructure than LangGraph or Agno.

### OpenAI Agents SDK

OpenAI's Agents SDK launched in March 2025 as a production-grade replacement for their experimental Swarm framework. The key abstraction is the handoff: agents transfer control to each other explicitly, carrying conversation context through the transition. Each agent declares its instructions, model, tools, and the list of agents it can hand off to. The runtime handles the routing.

This is a deliberately minimalist framework. Compared to LangGraph's typed-state graphs or CrewAI's role-based crews, the Agents SDK gives you very few primitives: agents, handoffs, tools, guardrails. The bet is that with good models, simple primitives compose into rich behavior, and that the framework should not get in the way. It's tightly integrated with OpenAI's models (though it supports other providers), and it ships with good tracing and observability out of the box.

It's a strong choice if you're already on OpenAI infrastructure and you want a clean, opinionated, production-friendly API. It's not a strong choice if you need cross-provider portability or sophisticated control flow.

### Claude Agent SDK and deepagents

Claude Agent SDK (Anthropic, 2025) and LangChain's deepagents are the clearest examples of the agent-harness philosophy. The frame is: don't try to encode the control flow at all. Instead, give the agent a strong system prompt, a curated set of tools, a file system as scratch memory, and a planning scaffold. Then run the LLM in a loop and let it drive. Sub-agents are spawned on demand by the main agent, not pre-wired by the developer.

The reason this works now and didn't in 2023 is that Claude 4 and GPT-5 are dramatically better at long-horizon planning, tool selection, and self-correction than their predecessors. The framework's job has shifted from "scaffold the model into reliability" to "remove obstacles so the model can be reliable on its own." File-system-as-memory, in particular, is a clever trick: instead of stuffing everything into the context window, the agent reads and writes files, which lets it work over arbitrarily long horizons without context pressure.

This is the most "bullish on models" position in the field, and it's the right philosophy for agent tasks where you genuinely don't know what control flow is needed in advance: open-ended research, code generation, complex debugging. It's the wrong philosophy when you do know the control flow, because hard-coding it is cheaper, more debuggable, and more reliable.

### Strands Agents

AWS shipped Strands Agents in 2025 as their entry. It's another harness-style framework, model-driven, with tight integration into Bedrock and AWS observability. The differentiator is enterprise integration: AWS's bet is that large companies want their agents inside the same IAM, logging, and compliance fabric they already run, and Strands provides that natively. Architecturally it's similar to OpenAI's Agents SDK and Claude Agent SDK; the value is in the integration surface, not in a novel abstraction.

### Google ADK

Google's Agent Development Kit, released in April 2025, takes a hierarchical-tree approach: a root agent delegates to sub-agents, which can have their own sub-agents. It's tightly integrated with Vertex AI and Gemini and ships with native A2A (Agent-to-Agent) protocol support, which lets ADK agents discover and invoke agents built with other frameworks. ADK is also distinctive for its native multimodal support, since it inherits Gemini's ability to process images, audio, and video as first-class inputs to an agent, which most other frameworks treat as an afterthought.

### Pydantic AI, smolagents, LlamaIndex, Semantic Kernel

These are the rest of the landscape worth naming. Pydantic AI, from the team behind the Pydantic validation library, is a Python framework focused on type safety and dependency injection, with a clean ergonomic API. It's a strong choice for type-strict Python codebases. Smolagents, from HuggingFace, is a deliberately minimalist library with a distinctive twist: agents think in code. Instead of producing JSON tool calls, the agent generates Python code that gets executed in a sandbox, which turns out to be significantly more expressive for many tasks. This is the same insight Anthropic has been writing about under the name "code execution with MCP."

LlamaIndex started as a RAG framework and has expanded into agents while keeping retrieval as its center of gravity. If your problem is fundamentally about getting the right information into the model, LlamaIndex's retrieval primitives are still best in class. Semantic Kernel is Microsoft's enterprise-oriented framework, especially strong in the .NET ecosystem and tightly integrated with Azure. It's the right pick if you're deploying inside a Microsoft enterprise stack.

## The protocol layer underneath all of this

The most important thing happening in the agent landscape, more important than any individual framework, is the emergence of protocols that sit beneath the frameworks. The two that matter are MCP and A2A.

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

MCP, the Model Context Protocol, was introduced by Anthropic in November 2024 to solve what they called the N×M integration problem: every model times every tool times every data source equals a custom integration to write. MCP defines a standard JSON-RPC protocol between an MCP client (your AI application) and an MCP server (a tool or data source), so a tool implemented once works with any MCP-compatible client. The original spec defined three primitives (tools, resources, prompts) and has since expanded to five, with sampling and roots added.

The adoption curve was unusually fast for a standard. By March 2025, OpenAI added MCP support to ChatGPT. By Q3 2025, Microsoft shipped MCP servers for GitHub, Azure, and Microsoft 365. By Q1 2026, Google added MCP to Gemini and Vertex AI. In December 2025, Anthropic donated MCP to the Linux Foundation's new Agentic AI Foundation, jointly governed with Block, OpenAI, AWS, Google, and Microsoft, which removed the last political reason for non-Anthropic players to resist it. The public registry now lists thousands of servers. SDK downloads cross 97 million per month. It is, at this point, the de facto integration standard.

The practical effect on agent development is enormous. You don't write custom integrations for Slack, GitHub, Postgres, or Google Drive anymore; you point your agent at the appropriate MCP server. Frameworks compete on agent control flow, observability, and developer experience, while the integration layer becomes commodity infrastructure. This is exactly what happened with HTTP for the web and JDBC for databases.

A2A, Agent-to-Agent, is the complement. Where MCP lets an agent talk to tools and data, A2A lets agents talk to other agents across framework boundaries. A LangGraph agent can invoke a CrewAI agent through A2A's standardized task interface. Adoption is still earlier than MCP but trending similarly, and Google's ADK ships with native support.

A second important development at this layer is what Anthropic has been calling "code execution with MCP" and what smolagents has done from the start. The insight is that as the number of available tools grows into the hundreds or thousands, loading every tool definition into the context window becomes prohibitively expensive in tokens. Instead, you expose tools as code on a virtual filesystem, give the agent a search-tools function and a code execution sandbox, and let it pull in only the tools it needs. The agent generates and runs a small script that calls the tools, instead of producing JSON for each tool call. This compresses both the input (tool definitions) and the intermediate state (tool outputs flow through code variables, not through the model's context), and it's how production agents at the high end will increasingly work.

## Where things actually are right now, May 2026

A few generalizations are safe at this point. First, the field has consolidated around a small number of serious players, all backed by frontier labs or hyperscalers. The four most-cited in 2026 framework comparisons are LangGraph, Claude Agent SDK, OpenAI Agents SDK, and Strands Agents, with CrewAI as the standout role-based alternative and Agno as the standout pure-Python performance play. Almost everything else either fits into one of those mental models or is a vertical specialization (LlamaIndex for retrieval, smolagents for code-thinking, OpenHands and Aider for coding agents specifically).

Second, the harness philosophy is clearly winning at the high-capability end. The cleanest evidence is that LangChain itself is now shipping deepagents, an explicitly harness-style API on top of langgraph, while keeping the graph layer underneath for cases where you do need it. As models get better, more orchestration moves into the model and out of the framework. The question for any framework is what value it adds beyond "thin wrapper around the model with good ergonomics."

Third, MCP has won as the integration standard. If you are designing a new agent system today, you should assume MCP is your integration layer, not an option you might add later. This frees you to pick a framework based on its control flow and developer experience, because the tool ecosystem is largely portable.

Fourth, the multi-agent fantasy from 2023 ("agents will collaborate emergently") has been mostly chastened by reality. The hard-won lesson is that most problems people thought needed multi-agent systems are better served by a single well-equipped agent with good tools and clear instructions. Multi-agent makes sense when you have genuinely independent specializations (a planner agent and an executor agent with different system prompts, or a generator and a critic), or when you need parallel exploration. It does not make sense as the default architecture for routine tasks. The 80/20 rule that LangChain itself now publishes is that 80% of real applications work better as a single agent.

Fifth, observability has become a distinct concern with its own market. LangSmith from LangChain, Langfuse, Helicone, Arize, and others are all competing on the "what did my agent actually do" problem. As production deployments scale, the ability to trace every model call, tool invocation, and state transition becomes operationally critical. The frameworks themselves are becoming OpenTelemetry-emitting so that you can plug them into whatever observability stack you already use.

Sixth, the "agent" buzzword has started to fragment usefully into more specific things: coding agents (Claude Code, Cursor, OpenHands, Cline, Aider), browser agents (Claude in Chrome, browser-use), workflow agents (n8n, Make), research agents (deepagents, Claude's research mode), and customer-facing agents (everything from support bots to phone agents). The general-purpose agent framework matters less when the agent is specialized; the specialized harness wins.

Seventh, there is a real and growing concern about agent governance and safety in production. Most enterprises deploying agents at scale are now investing in approval workflows, guardrails, audit logs, and human-in-the-loop checkpoints, often using third-party tooling rather than building it themselves. The frameworks that succeed in enterprise are the ones that take this seriously: LangGraph's checkpointing and human-in-the-loop primitives, Agno's runtime governance, n8n's hybrid deterministic-plus-AI design.

## How to actually choose

The honest answer is that you usually don't pick one. Real systems combine multiple frameworks: a LangGraph control flow calling MCP tools, embedded inside an n8n workflow for the integration plumbing, with observability through LangSmith or OpenTelemetry. The frameworks are not competing for the same slot in your stack; they're competing for slots in different slots.

That said, if forced to pick a single axis: pick LangGraph if you want maximum control and your problem has a real, knowable control flow structure; pick Agno if you want pure-Python ergonomics, fast iteration, and stateless scaling; pick Claude Agent SDK or deepagents if your problem is open-ended and you want to bet on the model; pick CrewAI if you have a team-of-specialists mental model and want to ship a prototype this week; pick n8n if your problem is mostly deterministic plumbing with AI in a few spots and a visual builder helps you reason about it; pick OpenAI Agents SDK if you're already deep in OpenAI's stack and want the path of least resistance. Skip AutoGen unless you're doing research or specifically need multi-agent dialogue.

Whatever you pick, build to MCP at the integration boundary, instrument with OpenTelemetry-compatible tracing, and design with the assumption that the model under your agent will be replaced with a better one in twelve months. The frameworks that will look smart in retrospect are the ones that treated themselves as scaffolding around the model, not as substitutes for it.
