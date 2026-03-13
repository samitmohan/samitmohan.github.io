---
layout: page
title: "Projects"
permalink: /projects/
---

This also consists of my upcoming projects (at the bottom; will keep updating as I make them <3)

<div class="project-grid has-featured">

  <div class="project-card featured" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="https://samitmohan.github.io/minitorch/">minitorch</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">PyTorch from scratch - tensors, autograd, and backpropagation implemented by hand to understand how deep learning frameworks actually work under the hood.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card featured" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/rag-for-blogs">personal-rag</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">RAG pipeline over my blog posts - chunking, embeddings, vector search, and retrieval-augmented generation for semantic blog search.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card featured" data-tag="tools">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/minicode">mini-code</a></div>
      <span class="project-tag tag-tools">tools</span>
    </div>
    <div class="project-description">Claude Code-style CLI tool - agentic coding assistant with file editing, terminal commands, and context management.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/deep-residual-learning-pytorch">resnet implementation</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Clean reimplementation of residual networks with skip connections, batch norm, and bottleneck blocks following the original paper.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="cv">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/tennis-analysis/tree/master">tennis analyser</a></div>
      <span class="project-tag tag-cv">cv</span>
    </div>
    <div class="project-description">Real-time tennis match analysis using object detection and tracking - player positions, ball trajectory, and court mapping.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="tools">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/spotify-mcp">spotify mcp server</a></div>
      <span class="project-tag tag-tools">tools</span>
    </div>
    <div class="project-description">MCP server that lets Claude analyse Spotify playlists, track listening patterns, and generate insights from your music data.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/resume-parser">resume parser</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">NLP-based resume screening system for ML roles - extracts skills, experience, and education to rank candidates before the GPT era.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="tools">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/Automated-Journal">python journal</a></div>
      <span class="project-tag tag-tools">tools</span>
    </div>
    <div class="project-description">Automated daily journal that prompts, timestamps, and archives entries - because writing should be frictionless.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="dsa">
    <div class="project-card-header">
      <div class="project-title"><a href="https://samitmohan.github.io/interviews/">dsa algorithms</a></div>
      <span class="project-tag tag-dsa">dsa</span>
    </div>
    <div class="project-description">Documentation of interview prep - data structures, algorithms, and problem-solving patterns.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/ML">ml algorithms</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">ML algorithms implemented from scratch - linear regression, SVMs, decision trees, k-means, and neural nets without frameworks.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="upcoming-divider">upcoming</div>

  <div class="project-card upcoming" data-tag="cuda">
    <div class="project-card-header">
      <div class="project-title"><a href="#">flash attention kernel</a></div>
      <span class="project-tag tag-cuda">cuda</span>
    </div>
    <div class="project-description">FlashAttention in Triton and CUDA - tiled, memory-efficient attention that avoids materializing the full N x N matrix. Optimizes the core bottleneck of transformers.</div>
    <div class="project-lang">Triton / CUDA</div>
  </div>

  <div class="project-card upcoming" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="#">mini-gpt</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">GPT-2 style language model built from scratch - tokenizer, multi-head attention, training loop, and text generation. A full decoder-only transformer pipeline.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="cuda">
    <div class="project-card-header">
      <div class="project-title"><a href="#">transformers in CUDA</a></div>
      <span class="project-tag tag-cuda">cuda</span>
    </div>
    <div class="project-description">Transformer architecture implemented entirely in CUDA from scratch - matrix multiplies, softmax, layer norm, and attention as custom kernels.</div>
    <div class="project-lang">CUDA / C++</div>
  </div>

  <div class="project-card upcoming" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="#">llama2</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Llama 2 from scratch - RoPE embeddings, grouped-query attention, RMSNorm, and SwiGLU. Full implementation of Meta's open-source LLM architecture.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="tools">
    <div class="project-card-header">
      <div class="project-title"><a href="#">autoresearch</a></div>
      <span class="project-tag tag-tools">tools</span>
    </div>
    <div class="project-description">Reimplementation of Karpathy's automated research assistant - paper discovery, summarization, and knowledge extraction pipeline.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="#">llmcouncil</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Multi-LLM debate and consensus system - multiple models discuss, critique, and converge on answers through structured deliberation.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="#">stable diffusion</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Stable diffusion paper implementation - UNet, noise scheduler, and denoising diffusion on MNIST. Understanding image generation from first principles.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="cv">
    <div class="project-card-header">
      <div class="project-title"><a href="#">multimodal vision</a></div>
      <span class="project-tag tag-cv">cv</span>
    </div>
    <div class="project-description">Multimodal vision project - bridging text and image understanding with cross-modal attention and representation learning.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="tools">
    <div class="project-card-header">
      <div class="project-title"><a href="#">shelfml</a></div>
      <span class="project-tag tag-tools">tools</span>
    </div>
    <div class="project-description">Book-keeping ML app with a FastAPI backend - track reading lists, recommendations, and reading patterns powered by machine learning.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="#">sql-model</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Train a small language model to write SQL - fine-tune on text-to-SQL datasets to generate queries from natural language.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="#">ai chatbot</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Conversational AI chatbot with retrieval, memory, and tool use - end-to-end from embeddings to streaming responses.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card upcoming" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="#">shazam</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Recreating Shazam's audio fingerprinting with deep learning - spectrograms, convolutional features, and nearest-neighbor matching for song identification.</div>
    <div class="project-lang">Python</div>
  </div>

</div>
