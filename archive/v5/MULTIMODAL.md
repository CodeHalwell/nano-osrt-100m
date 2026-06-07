# Multimodal Extension Notes — NanoOSRT v5

> **Status:** notes / pre-design. Not implemented. Captures the
> architectural options, trade-offs, and the recommended path if we
> were to add image input to nano-osrt. Audio and video are out of
> scope for these notes.

---

## 1. The fundamental question

How does an image get into a language model? The answer determines
almost every other design choice. Three real options, in increasing
complexity:

1. **One vector per image** — encode the whole image into a single
   embedding, prepend it as one "token" before the text.
2. **A sequence of patch tokens per image** — encode the image into N
   embeddings (e.g. 256–576), each describing a spatial region, and
   splice them into the text sequence at an `<|image|>` placeholder.
3. **Cross-attention layers** — keep the text stream untouched, add
   dedicated cross-attention blocks in the LM that read from a frozen
   image-feature memory (Flamingo-style).

**Modern dominant choice: option 2.** Why:
- Option 1 destroys spatial information — the model can't answer
  "what's in the bottom-left?" or count items.
- Option 3 (cross-attention) is more parameter-efficient but requires
  surgically inserting new layers into an existing LM, which fights
  the recursive-block design.
- Option 2 — image tokens that look identical in shape to text tokens
  and share the LM's existing attention, MLP, MoE, and RoPE pathways
  — is the LLaVA / Qwen-VL / InternVL / MoE-LLaVA recipe.

The rest of these notes assume option 2.

---

## 2. CNN vs ViT for the encoder

The user's question: *can we use a CNN to analyse the image and
flatten to a vector embedding, then feed through KQV projections?*

### 2.1 The "flatten to a vector" framing — careful here

There are two different "flatten" operations that get conflated:

**Bad flatten — global pooling to a single vector:**
```
image (224×224×3)
  → CNN
  → feature map (7×7×D)
  → global avg pool
  → single vector (1×D)
  → KQV projection
  → one token in the LM sequence
```
This is option 1 from §1. Throws away spatial information. Quality
drops 20-40% on benchmarks like GQA / VQA / MMBench compared to
patch-token approaches. **Don't do this.**

**Good flatten — keep spatial axis as sequence:**
```
image (224×224×3)
  → CNN or ViT
  → feature map (H×W×D)         e.g. 7×7×768 or 14×14×1024
  → flatten H×W → sequence axis
  → sequence of N patch embeddings (N×D)
  → MLP projector (D → 1536)
  → N image tokens in the LM sequence
```
This is option 2. The "flatten" here means "treat the spatial grid as
a 1D sequence of N tokens", not "compress to one vector". Each
spatial location becomes its own token; the LM's self-attention
re-establishes spatial relationships across the patch tokens.

### 2.2 CNN vs ViT for producing those patch features

Either works. The choice is mostly determined by what's pre-trained:

| Encoder | Patch token count | Hidden dim | Pretraining | Notes |
|---------|------------------:|-----------:|-------------|-------|
| **SigLIP-So400M** (ViT) | 256–729 | 1152 | 10B image-text pairs | Default for modern VLMs (LLaVA-NeXT, Qwen2-VL) |
| **CLIP-ViT-L/14** | 256 | 1024 | 400M pairs | LLaVA-1.5 default |
| **DINOv2-L** (ViT) | 256 | 1024 | 142M images, no captions | Strong dense features, weaker text alignment |
| **ConvNeXt-L** (CNN) | 49–256 (depends on resolution) | 1024–1536 | ImageNet-21k or LAION | Modern CNN; comparable to ViT-L on perception |
| **EfficientNet-V2** (CNN) | configurable | 1280–1408 | ImageNet | Older but light |

**The CNN-vs-ViT decision in 2026 is mostly a non-issue:** ViTs
dominate VLM stacks because they were trained on the largest
image-text contrastive datasets (CLIP, SigLIP), and pretraining scale
beats architecture choice. A pretrained ConvNeXt is fine; a CNN
trained from scratch is not — there's no reason to throw away decades
of pretraining work.

**Recommendation:** SigLIP-So400M. Its output is already shaped as
patch tokens — no special CNN→sequence conversion needed.

### 2.3 The KQV question

> "feed through KQV projections etc."

Two places KQV happens:

1. **Inside the encoder.** A ViT runs its own attention internally
   over patch tokens before they reach the LM. A CNN does not — its
   features come out of conv layers without attention. If you used a
   CNN you'd get (B, H, W, D) features straight from conv stages, no
   internal attention.
2. **Inside the LM after projection.** Once image features are
   projected to `dim=1536` and spliced into the LM's input embedding
   sequence, they go through the LM's attention layers exactly like
   text tokens. The same Q/K/V weights process both modalities. **The
   image tokens go through the LM's MoE routing, attention, RoPE, and
   recursive loops with no special-case code.** This is the
   architectural simplicity that makes option 2 popular.

So yes — *after* the encoder, image embeddings ride the same KQV
projections as word embeddings. That's the point. Whether the
encoder itself uses KQV (ViT does, CNN doesn't) is a separate
question, and it doesn't matter much because the LM will re-process
the features regardless.

---

## 3. Feed embeddings as if they were word tokens

> "Should the encoder feed embeddings into the model as if they were
> word tokens?"

**Yes.** This is the single most important property of the LLaVA-style
recipe. After the encoder + projector, image tokens are
indistinguishable in shape from text tokens:

```
text token   : embedding ∈ ℝ^1536
image token  : embedding ∈ ℝ^1536   ← same shape!
```

They share:
- The same `hidden_dim` (1536).
- The same attention layers (image patches attend to each other and
  to text; text attends back to image).
- The same MoE routing (each image token picks 2 of 8 experts; routes
  may specialise toward visual features over training).
- The same RoPE positional encoding (sequence position; see §5.2 for
  whether 2D RoPE is needed).
- The same recursive loops (image tokens get 18 effective layers of
  processing, just like text).

What's *different* is purely how they enter the model:

- Text tokens: `text_id → embed_tokens[id] → ℝ^1536`
- Image tokens: `pixel_values → encoder → patch_features → projector → ℝ^1536`

The forward pass becomes:

```python
def forward(input_ids, pixel_values=None, ...):
    text_embeds = self.embed_tokens(input_ids)             # (B, T, 1536)

    if pixel_values is not None:
        image_features = self.vision_encoder(pixel_values) # (B, P, 1024)
        image_embeds = self.projector(image_features)      # (B, P, 1536)
        # Splice image_embeds into text_embeds at <|image|> positions:
        embeds = splice_at_image_placeholders(
            text_embeds, image_embeds, input_ids, self.config.image_token_id,
        )
    else:
        embeds = text_embeds

    # Everything from here is unchanged — recursive blocks, MoE,
    # attention, output projection. The image tokens just ride along.
    return self.lm_head(self.run_blocks(embeds))
```

The splice is the only new operation. Everything downstream is
identical to text-only inference.

---

## 4. nano-osrt-specific design notes

### 4.1 What nano-osrt brings to multimodality (good news)

1. **MoE routes can specialise per modality without code changes.**
   Image patches and text tokens will route to different experts
   within the same 8-expert pool over training. This is observed in
   MoE-LLaVA — "vision experts" emerge naturally. For nano-osrt's
   3 blocks × 8 experts = 24 routed experts, there's plenty of room.
2. **Recursive depth is free.** Image patches get 18 effective
   transformer layers without inflating parameter count. Most small
   VLMs are bottlenecked on visual reasoning depth. Recursion
   addresses that at zero parameter cost.
3. **HRA adapters already in place.** Visual instruction tuning can
   train the projector + HRA only, leaving the pretrained LM weights
   frozen. Standard trick to prevent text-capability regression.
4. **Native single-token tags already supported.** The tokenizer has
   unused FIM token slots that can be repurposed for `<|image|>` and
   `<|/image|>` without retokenization.

### 4.2 What complicates things (open risks)

1. **Recursion + image tokens is untested.** No existing VLM uses
   recursive transformer blocks. The loop adapters (rank-16) might
   need re-tuning if they over-fit to text-only patterns. Possible
   mitigation: freeze loop adapters during VL-1 alignment, unfreeze
   only during VL-2.
2. **MoE balance loss could be skewed by image-vs-text token ratio.**
   At 576 image tokens vs ~512 text tokens per typical VLM example,
   the load-balancing loss sees mostly image tokens. The Switch
   balance objective naturally adapts (it just balances the
   distribution it's given), but the bias controller's update rate
   (0.10) was tuned for text-only mixes. Worth re-checking.
3. **Position encoding (RoPE) is 1D only.** Image patches inherit
   sequence-position RoPE, not 2D spatial RoPE. Most VLMs do this and
   it works fine, but it leaves spatial reasoning slightly
   on-the-table vs Qwen2-VL's M-RoPE. Cheap fix is available later.
4. **The base LM is small (363M).** Multimodal quality scales with
   text base. nano-osrt-VL would land in the LLaVA-Phi / MoE-LLaVA
   weight class — competitive at sub-7B but below LLaVA-1.5-7B. The
   recursive MoE provides some compensation; how much is empirical.

### 4.3 Sequence length budget

At seq 8192, with each image taking ~576 patch tokens:

| Use case | Image tokens | Text budget left |
|----------|-------------:|-----------------:|
| 1 image + conversation | 576 | 7616 (~6000 words) |
| 4 images + Q&A | 2304 | 5888 |
| Document understanding (8 pages as images) | 4608 | 3584 |
| Video (16 frames at 144 patches each) | 2304 | 5888 |

This is generous. The 8192 context was originally chosen for text-only
long-form reasoning; it scales gracefully to multi-image input
without architectural changes.

---

## 5. Recommended training pipeline

Two new stages on top of the existing text pipeline. Both reuse the
HRA infrastructure already shipped — train HRA + projector, keep base
weights frozen.

### 5.1 Stage VL-1: vision–language alignment

**Goal:** teach the projector to put image features in a place the LM
can read. The LM and vision encoder both stay frozen.

| | |
|--|--|
| Trainable params | Projector only (~5 M) |
| Frozen | Base LM (363 M) + HRA + vision encoder (~400 M) |
| Datasets | CC3M, BLIP-pretrain, LLaVA-Pretrain (558 K image-caption pairs total) |
| Loss | Standard next-token LM loss on the caption, image tokens contribute zero loss |
| Steps / cost | ~1 epoch, ~50 H100-hours, ~$200 |

After VL-1 the model can produce coherent captions but doesn't yet
follow visual instructions well.

### 5.2 Stage VL-2: visual instruction tuning

**Goal:** teach the model to *use* image content for instruction
following. HRA + projector trainable; base LM still frozen.

| | |
|--|--|
| Trainable params | Projector (~5 M) + HRA (~86 M) |
| Frozen | Base LM (363 M) + vision encoder |
| Datasets | LLaVA-Instruct-665K, ShareGPT4V, M3IT |
| Steps / cost | ~1-2 epochs, ~30 H100-hours, ~$120 |

Total VL training cost: ~$300-500.

### 5.3 Optional Stage VL-3: full unfreezing

If quality after VL-2 is short of target, unfreeze the full base LM
for one short cooling pass on visual instruction data. Risks
text-regression — must monitor text-only eval loss as a guard.

---

## 6. Implementation plan (engineering scope)

### 6.1 New files

```
src/nano_osrt/
  vision_encoder.py    # SigLIP wrapper + image preprocessing
  multimodal.py        # Projector + splice logic
  multimodal_data.py   # Multimodal SFT stream (image+text packing)
```

### 6.2 Touched files

```
src/nano_osrt/
  config.py            # vision_dim, image_token_id, projector_hidden
  model.py             # forward(pixel_values=...), splice hook
  train_config.py      # VLMConfig, VLMInstructConfig
app.py                 # --stage vl_align, --stage vl_instruct
```

### 6.3 Engineering effort estimate

| Component | Lines | Days |
|-----------|------:|-----:|
| `vision_encoder.py` (HF wrapper, preprocessor) | ~80 | 1 |
| `multimodal.py` (projector + splice) | ~150 | 2 |
| `multimodal_data.py` (data pipeline) | ~250 | 3-4 |
| `model.py` integration (forward hook) | ~50 | 1 |
| Config + app.py wiring | ~100 | 1 |
| Tests (multimodal forward, splice correctness, batched inference) | ~200 | 2 |
| Eval harness (MMBench, MMMU-mini, GQA) | ~150 | 2-3 |
| **Total** | **~1000** | **~12-15 days** |

Plus ~$300-500 cloud spend for VL-1 + VL-2.

---

## 7. Open questions to resolve before starting

1. **Vision encoder choice.** SigLIP-So400M (default) vs DINOv2-L (no
   text alignment but stronger dense features) vs CLIP-ViT-L
   (smaller, well-trodden). SigLIP wins by default, but DINOv2 is
   worth A/B if we want stronger spatial/perceptual ability.
2. **Image resolution.** 224×224 (256 patches) vs 384×384 (576
   patches) vs LLaVA-NeXT-style multi-resolution (up to 2880 patches).
   Higher resolution helps OCR and fine detail but eats sequence
   length. Probably 384 is the sweet spot.
3. **Single-image or multi-image from day one.** Multi-image needs
   extra `<|image|>` placeholder handling but no architectural change.
   Worth supporting from the start.
4. **Should HRA cover the vision pathway?** Currently HRA wraps
   `linear` layers in the LM. The projector is a small MLP — could
   either skip HRA there (project direct, train normally) or extend
   HRA to wrap the projector for consistency. Probably skip — the
   projector is small enough to train directly without interfering
   with base LM stability.
5. **Position encoding.** Stick with 1D sequence-position RoPE for
   simplicity, or implement 2D (M-RoPE) for better spatial awareness?
   1D is the safe default; 2D is a v2 improvement.
6. **Routing telemetry.** The existing W&B MoE dashboard
   (`moe/clean_per_token_entropy_mean`,
   `moe/clean_assignment_entropy_mean`) would benefit from a
   per-modality breakdown — track entropy separately for image
   tokens vs text tokens so we can directly observe vision experts
   emerging.

---

## 8. Bottom line

Yes, multimodality is feasible — and nano-osrt's recursive MoE may
actually be a *good* fit for it (vision experts can emerge naturally,
recursion gives free depth, HRA enables safe instruction tuning).

The work is roughly **2-3 weeks of code + ~$300-500 of compute**. The
biggest risks are (a) recursion + vision is untested, and (b) the
small text base limits quality vs 7B-class VLMs. Neither is a
blocker.

If/when this becomes a real project, the natural cut point is after
GRPO completes: the text model will have stabilised, evaluation
infrastructure will be in place, and the codebase has the HRA + MoE
plumbing the multimodal recipe needs.
