# Dataset Reference for nano-osrt-100m

Curated list of high-quality datasets for LLM training. Datasets should target
three characteristics: **accuracy**, **diversity**, and **complexity**.

## Currently Used in Pipeline

### Math SFT (Stage 2: pretrain -> **SFT**)
| Dataset | Weight | HF ID | Notes |
|---------|--------|-------|-------|
| GSM8K | 25% | `openai/gsm8k` (config: `main`) | Grade school math, gold standard |
| Orca-Math | 25% | `microsoft/orca-math-word-problems-200k` | GPT4-Turbo generated word problems |
| NuminaMath-CoT | 20% | `AI-MO/NuminaMath-CoT` | 859k samples, AI Math Olympiad winner |
| MathInstruct | 15% | `TIGER-Lab/MathInstruct` | Diverse math reasoning |
| LongForm | 15% | `akoksal/LongForm` | Long-form responses for diversity |

### GRPO (Stage 3: SFT -> **GRPO**)
| Dataset | HF ID | Notes |
|---------|-------|-------|
| GSM8K | `openai/gsm8k` (config: `main`) | Prompts only, verifiable rewards |

### General SFT (Stage 4: GRPO -> **General SFT**)
| Dataset | Weight | HF ID | Notes |
|---------|--------|-------|-------|
| Alpaca Cleaned | 25% | `yahma/alpaca-cleaned` | General instructions |
| OpenHermes 2.5 | 20% | `teknium/OpenHermes-2.5` | High-quality conversations |
| SlimOrca | 20% | `Open-Orca/SlimOrca-Dedup` | Deduplicated Orca data |
| IFEval-like | 10% | `argilla/ifeval-like-data` (config: `filtered`) | Instruction following |
| LongForm | 15% | `akoksal/LongForm` | Long-form diversity |
| GSM8K | 10% | `openai/gsm8k` (config: `main`) | Math retention |

---

## Candidate Datasets (Not Yet Used)

Organized by category. All permissive licenses unless noted.

### General Instruction

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| Nemotron-Post-Training-Dataset-v2 | 6.34M | Nvidia | Aug 2025 | Multilingual (ES, FR, DE, IT, JA), math/code/reasoning. Used for Nemotron-Nano-9B-v2. |
| smoltalk2 | 3.38M | Hugging Face | Jul 2025 | Used for SmolLM3. Includes OpenThoughts3, Tulu 3, multilingual data. |
| open-perfectblend | 1.42M | Xu et al., Labonne | Oct 2024 | Solid general-purpose mix: chat, math, code, instruction following. |
| orca-agentinstruct-1M-v1 | 1.05M | Microsoft | Nov 2024 | Subset of AgentInstruct, web seed data. |
| tulu3-sft-mixture | 939k | AllenAI | Nov 2024 | CC-BY-NC-4.0. SFT mixture for Tulu 3, includes persona-based answers. |
| FuseChat-Mixture | 95k | Wan et al. | Feb 2024 | Diverse styles, human + model generated. |

### Math

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| MegaScience | 1.25M | GAIR-NLP | Jul 2025 | CC-BY-NC-SA-4.0. Scientific domains with ablation studies. |
| OpenThoughts3-1.2M | 1.2M | OpenThoughts | Jun 2024 | 850k math, 250k code, 100k science. Annotated with QwQ-32B. |
| NuminaMath-CoT | 859k | Jia Li et al. | Jul 2024 | **USED IN PIPELINE.** AI Math Olympiad winner. |
| AM-Thinking-v1-Distilled (Math) | 558k | a-m-team | May 2025 | Verified responses distilled from AM-Thinking-v1 and Qwen3-235B. |
| OmniThought-0528 | 365k | Alibaba-PAI | Jun 2025 | Math/code/science distilled from DeepSeek-R1 and QwQ-32B. |
| Orca-Math | 200k | Mitra et al. | Feb 2024 | **USED IN PIPELINE.** Grade school math from GPT4-Turbo. |

### Code

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| Ling-Coder-SFT | 4.48M | InclusionAI | Mar 2025 | 20 programming languages, EN/ZH. |
| rStar-Coder | 1M | Microsoft | May 2025 | Competitive code problems. |
| opc-sft-stage2 | 436k | Huang et al. | Nov 2024 | OpenCoder Stage 2 data. |
| AM-Thinking-v1-Distilled (Code) | 324k | a-m-team | May 2025 | Verified code responses. |
| CodeFeedback-Filtered-Instruction | 157k | Zheng et al. | Feb 2024 | Filtered Magicoder + ShareGPT. |
| synthetic_tex_to_sql | 100k | Gretel.ai | Apr 2024 | Text-to-SQL across diverse domains. |

### Instruction Following

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| AutoIF-instruct-61k | 61.5k | Diao et al. | Oct 2024 | Generated with gpt-4o-mini via Qwen's AutoIF. |
| ifeval-like-data | 56.3k | Argilla | Oct 2024 | **USED IN PIPELINE** (filtered subset). Qwen2.5-72B generated, verified. |
| tulu-3-sft-personas-IF | 30k | AllenAI | Nov 2024 | Persona-based synthetic IF data. |

### Multilingual

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| luth-sft | 570k | kurakurai | Aug 2025 | French/English with good curation. |
| aya dataset | 204k | Singh et al. | Feb 2024 | Community-curated multilingual IF. |
| M2Lingual | 175k | ServiceNow AI | Jun 2024 | 70+ languages, 20 NLP tasks. |

### Agent & Function Calling

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| xlam-function-calling-60k | 60k | Salesforce | Jun 2024 | Verifiable function-calling data. |
| FunReason-MT | 17k | Hao et al. | Oct 2025 | Multi-turn function calling with CoT. |
| hermes-function-calling-v1 | 11.6k | Nous | Aug 2024 | Structured output and function calling. |
| ToolACE | 11.3k | Liu et al. | Aug 2024 | Self-evolution synthesized API pool. |
| APIGen-MT-5k | 5k | Salesforce | Apr 2025 | CC-BY-NC-4.0. Multi-turn agentic trajectories. |

### Real Conversations

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| WildChat-4.8M | 3.2M | Allen AI | Aug 2025 | Non-toxic human-ChatGPT conversations. |
| lmsys-chat-1m | 1M | LMSYS | Sep 2023 | Real conversations with 25 LLMs. |
| arena-human-preference-100k | 110k | LMSYS | Feb 2025 | Human preference evaluations from Chatbot Arena. |

### Preference (DPO/ORPO)

| Dataset | # Samples | Authors | Date | Notes |
|---------|-----------|---------|------|-------|
| Skywork-Reward-Preference-80K | 77k | Skywork | 2024 | Compiled from HelpSteer2, OffsetBias, WildGuard, Magpie. |
| ultrafeedback-binarized-cleaned | 61.1k | Argilla | 2023 | Decontaminated UltraChat, GPT-4 scored. |
| Infinity-Preference | 59k | BAAI | Sep 2024 | Task-weighted preference pairs. |
| Code-Preference-Pairs | 53k | Vezora | Jul 2024 | Correct vs buggy code pairs. |
| orpo-dpo-mix-40k | 44k | Argilla, Labonne | May 2024 | High-quality DPO compilation. |
| HelpSteer3 | 40.5k | Wang et al. | Oct 2024 | Multi-attribute, 14 languages. |
| chatbot_arena_conversations | 33k | LMSYS | Jul 2023 | Real pairwise preferences. |
| FalseReject | 28.8k | Amazon Science | May 2025 | CC-BY-NC-4.0. Mitigating over-refusal. |
| tulu-3-pref-personas-IF | 19.9k | AllenAI | Nov 2024 | IF-specific preference data. |
| Human-Like-DPO-Dataset | 10.9k | Weyaxi | May 2024 | Less formal/robotic outputs. |

---

## Recommendations for Future Versions

### v4 (MoE) — Higher Capacity
With 12 experts and more effective parameters, these become viable:
- **OpenThoughts3-1.2M**: Rich math/code/science CoT, ideal for reasoning MoE
- **open-perfectblend**: Well-balanced general-purpose SFT mix
- **opc-sft-stage2**: Code capability (if adding code as a domain)

### v3.x — If Math SFT Plateaus
- **AM-Thinking-v1-Distilled (Math)**: Verified distilled reasoning, higher difficulty
- **OmniThought-0528**: Math + science from DeepSeek-R1/QwQ-32B

### Preference Alignment (Future DPO Stage)
If adding DPO after GRPO:
- **ultrafeedback-binarized-cleaned**: Best general-purpose preference data
- **orpo-dpo-mix-40k**: Curated high-quality DPO pairs
- **Human-Like-DPO-Dataset**: Reduces robotic output style

---

## Tools for Dataset Quality

### Filtering
- **SemHash**: Fuzzy deduplication via fast embeddings
- **Argilla**: Collaborative annotation and filtering
- **Rule-based**: Remove refusals ("As an AI assistant...")

### Generation
- **Curator**: Synthetic data pipelines around LLMs
- **Distilabel**: SFT/DPO augmentation (UltraFeedback, DEITA)
- **Augmentoolkit**: Raw text -> dataset conversion

### Exploration
- **Lilac**: Dataset exploration and quality control
- **Nomic Atlas**: Interactive embedding visualization
- **text-clustering**: HuggingFace text clustering framework
