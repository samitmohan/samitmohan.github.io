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
    <div class="project-description">PyTorch clone from scratch - reverse-mode autodiff, Module system (Linear, Conv2d, BatchNorm, Dropout), Adam/SGD optimizers, LR schedulers, and optional CUDA support via CuPy. Trains MNIST with both MLP and CNN.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card featured" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/rag-for-blogs">personal-rag</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">Hybrid RAG with dual-mode architecture: CPU-only retrieval on Render (FAISS + SentenceTransformers) for public use, local Qwen 2.5:7b via Ollama for synthesized answers. Citation system grounding answers in specific blog sections.</div>
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
    <div class="project-description">Faithful reproduction of the 2015 ResNet paper's CIFAR-10 experiments - trains plain vs residual nets at depths 20-110, with results closely matching original error rates. Includes ImageNet variants (ResNet-18 to 152).</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="cv">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/tennis-analysis/tree/master">tennis analyser</a></div>
      <span class="project-tag tag-cv">cv</span>
    </div>
    <div class="project-description">Multi-model CV pipeline: YOLOv8x for player tracking, fine-tuned YOLO for ball detection, ResNet50 for 14-keypoint court mapping. Calculates shot velocity, groups rallies, and generates per-player court heatmaps.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="tools">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/spotify-mcp">spotify mcp server</a></div>
      <span class="project-tag tag-tools">tools</span>
    </div>
    <div class="project-description">MCP server with 21 tools for natural-language Spotify control - playback, playlist management, music discovery, and lyrics retrieval via Genius API. Integrates with Claude Desktop.</div>
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
    <div class="project-description">CLI journaling tool with NLP-powered analysis - DistilBERT sentiment, 7-emotion detection, YAKE keyword extraction, streak tracking, morning/evening prompts, and a Streamlit + Plotly dashboard for mood trends.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="dsa">
    <div class="project-card-header">
      <div class="project-title"><a href="https://samitmohan.github.io/interviews/">dsa algorithms</a></div>
      <span class="project-tag tag-dsa">dsa</span>
    </div>
    <div class="project-description">400+ solutions across LeetCode, Neetcode 150, and Striver Sheet. 12 from-scratch data structure implementations, OS fundamentals, design patterns, and a time complexity cheatsheet.</div>
    <div class="project-lang">Python</div>
  </div>

  <div class="project-card" data-tag="ml">
    <div class="project-card-header">
      <div class="project-title"><a href="https://github.com/samitmohan/ML">ml algorithms</a></div>
      <span class="project-tag tag-ml">ml</span>
    </div>
    <div class="project-description">100+ educational implementations: MicroGPT (complete GPT in 218 lines), ResNet-34 three ways, attention mechanisms, CNNs, LSTMs, autograd engine, linear algebra, and optimization methods - all from scratch.</div>
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
