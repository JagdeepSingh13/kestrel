# Reading list

From the project handoff (sections 11, 12). Links may move; if one breaks, search the title.
Each `docs/learning/M<n>.md` lesson links the 1 to 2 items that matter for that step.

## Reading mapped to milestones

| Milestone | Read |
|---|---|
| Before M0 | Tier 1 items 1 to 4, mini-SWE-agent source |
| M1 to M2 | Codex agent loop post, smolagents, 12-Factor Agents, OpenHands event stream |
| M3 | Writing Effective Tools, Lethal Trifecta, Claude Agent SDK permissions |
| M4 | BFCL, Ollama tool support + structured outputs posts, GBNF, ReAct paper, SWE-agent paper |
| M5 to M6 | Context Engineering, Long-Running Agents, Aider repo map, Deep Agents |
| M7, M7.5 | Hamel evals post, Lilian Weng harness post, Meta-Harness paper |
| M8 to M9 | Multi-agent research post, Böckeler harness engineering, Agent SDK hooks |

Inner vs outer harness: much "harness engineering" writing is about the outer harness people build around coding agents (AGENTS.md, linters, CI). Kestrel is an inner harness (loop, tools, context, safety). Prefer inner-harness material.

## Harness-specific

### Tier 1, read in order
1. Anthropic, Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents
2. LangChain, Anatomy of an Agent Harness: https://www.langchain.com/blog/the-anatomy-of-an-agent-harness
3. Addy Osmani, Agent Harness Engineering: https://addyosmani.com/blog/agent-harness-engineering/
4. OpenAI, Unrolling the Codex Agent Loop: https://openai.com/index/unrolling-the-codex-agent-loop/
5. Anthropic, Effective Harnesses for Long-Running Agents: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
6. Anthropic, Effective Context Engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
7. Anthropic, Writing Effective Tools for Agents: https://www.anthropic.com/engineering/writing-tools-for-agents
8. 12-Factor Agents: https://github.com/humanlayer/12-factor-agents

### Tier 2, the discipline
- Lilian Weng, Harness Engineering for Self-Improvement: https://lilianweng.github.io/posts/2026-07-04-harness/
- OpenAI, Harness Engineering: https://openai.com/index/harness-engineering/
- Böckeler, Harness Engineering for Coding Agent Users: https://martinfowler.com/articles/harness-engineering.html
- Thoughtworks podcast: https://www.thoughtworks.com/insights/podcasts/technology-podcasts/what-harness-engineering
- Hidden Technical Debt: Agent Harness: https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/
- MongoDB, LLM Is the Smallest Part: https://mongodb.com/company/blog/technical/agent-harness-why-llm-is-smallest-part-of-your-agent-system
- Marmelab, State of Harness Engineering 2026: https://marmelab.com/blog/2026/09/24/the-state-of-ai-harness-engineering-2026.html
- Ralph loop: https://ghuntley.com/loop/
- Also search: Google Developers Blog "anatomy of harness engineering" (Sept 2026), "Meta-Harness" paper (March 2026).

### Tier 3, source code, smallest first
- mini-SWE-agent: https://github.com/SWE-agent/mini-swe-agent
- smolagents: https://github.com/huggingface/smolagents
- llm: https://github.com/simonw/llm
- Anthropic quickstarts: https://github.com/anthropics/claude-quickstarts
- Deep Agents: https://github.com/langchain-ai/deepagents
- Aider: https://github.com/Aider-AI/aider
- Goose: https://github.com/block/goose
- Codex CLI: https://github.com/openai/codex
- Gemini CLI: https://github.com/google-gemini/gemini-cli
- OpenHands: https://github.com/All-Hands-AI/OpenHands
- SWE-agent: https://github.com/SWE-agent/SWE-agent

## General

### Suggested schedule
- Week 1: phases 1 and 2. Install Ollama, run models, call the API from Python by hand.
- Week 2: phase 3. Write a 50-line agent loop yourself, no framework.
- Week 3: smolagents source + Hugging Face Agents Course. Kestrel M0 to M1.
- Week 4+: one milestone per week, with matching reading.

### Phase 1: how LLMs work
- Karpathy, Intro to Large Language Models: https://www.youtube.com/watch?v=zjkBMFhNj_g
- Karpathy, Deep Dive into LLMs like ChatGPT: https://www.youtube.com/watch?v=7xTGNNLPyMI
- Karpathy, Let's build GPT from scratch: https://www.youtube.com/watch?v=kCc8FmEb1nY
- 3Blue1Brown, Neural Networks: https://www.3blue1brown.com/topics/neural-networks
- Jay Alammar, The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Raschka, Build an LLM from Scratch: https://github.com/rasbt/LLMs-from-scratch

### Phase 2: running local models
- Ollama: https://github.com/ollama/ollama, docs https://docs.ollama.com, library https://ollama.com/library
- Ollama tool support: https://ollama.com/blog/tool-support
- Ollama structured outputs: https://ollama.com/blog/structured-outputs
- LM Studio docs: https://lmstudio.ai/docs
- Apple MLX: https://github.com/ml-explore/mlx, mlx-lm: https://github.com/ml-explore/mlx-lm
- llama.cpp: https://github.com/ggml-org/llama.cpp
- llama.cpp grammars (GBNF): https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- Visual Guide to Quantization: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
- Hugging Face models: https://huggingface.co/models
- r/LocalLLaMA: https://www.reddit.com/r/LocalLLaMA/

### Phase 3: agent concepts
- Anthropic, Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents
- Anthropic, Effective Context Engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, Writing Effective Tools for Agents: https://www.anthropic.com/engineering/writing-tools-for-agents
- Anthropic, Multi-Agent Research System: https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic engineering blog: https://www.anthropic.com/engineering
- OpenAI, Practical Guide to Building Agents: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Lilian Weng, LLM Powered Autonomous Agents: https://lilianweng.github.io/posts/2023-06-23-agent/
- Chip Huyen, Agents: https://huyenchip.com/2025/01/07/agents.html
- Chip Huyen, AI Engineering resources: https://github.com/chiphuyen/aie-book
- 12-Factor Agents: https://github.com/humanlayer/12-factor-agents
- Simon Willison's blog: https://simonwillison.net

### Phase 4: courses
- Hugging Face Agents Course: https://huggingface.co/learn/agents-course
- DeepLearning.AI short courses: https://www.deeplearning.ai/short-courses/
- MCP course with Anthropic: https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/
- Andrew Ng, Agentic AI: https://www.deeplearning.ai/courses/agentic-ai/
- Anthropic courses: https://github.com/anthropics/courses
- Anthropic Academy: https://anthropic.skilljar.com

### Phases 5 to 10: code, papers, MCP, evals, security, Python
- Agent code: smolagents, llm, Aider, Goose, OpenHands (links above), Claude Agent SDK https://docs.claude.com/en/api/agent-sdk/overview
- Papers: ReAct https://arxiv.org/abs/2210.03629, Toolformer https://arxiv.org/abs/2302.04761, Reflexion https://arxiv.org/abs/2303.11366, SWE-agent https://arxiv.org/abs/2405.15793, Gorilla https://arxiv.org/abs/2305.15334
- MCP: docs https://modelcontextprotocol.io, Python SDK https://github.com/modelcontextprotocol/python-sdk, servers https://github.com/modelcontextprotocol/servers
- Tool calling benchmark: Berkeley Function Calling Leaderboard https://gorilla.cs.berkeley.edu/leaderboard.html
- JSON Schema: https://json-schema.org/learn/getting-started-step-by-step
- Evals: Hamel Husain https://hamel.dev/blog/posts/evals/, Inspect AI https://inspect.aisi.org.uk, lm-evaluation-harness https://github.com/EleutherAI/lm-evaluation-harness, tau-bench https://github.com/sierra-research/tau-bench, SWE-bench https://www.swebench.com
- Security: Lethal Trifecta https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/, OWASP LLM Top 10 https://genai.owasp.org/llm-top-10/
- Python stack: uv https://docs.astral.sh/uv/, asyncio https://docs.python.org/3/library/asyncio.html, Pydantic https://docs.pydantic.dev, httpx https://www.python-httpx.org, Typer https://typer.tiangolo.com, pytest https://docs.pytest.org
- Newsletter/podcast: Latent Space https://www.latent.space
