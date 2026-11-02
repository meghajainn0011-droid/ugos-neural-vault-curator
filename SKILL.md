---
name: ugos-neural-vault-curator
version: 1.0.0
description: Autonomous on-device semantic memory indexing, privacy-preserving photo vault clustering, and creative temporal reel generator for UGREEN AI NAS (UGOS).
author: MD Shahid Anwar
license: MIT
tags: [ai-nas, digital-memories, semantic-search, workflow-automation, local-ai]
---

# UGOS Neural Vault & Media Curator

## 1. Overview
The **UGOS Neural Vault & Media Curator** is a lightweight, 100% on-device AI Skill engineered specifically for UGREEN AI NAS. It solves two critical consumer NAS problems with zero cloud dependency:

1. **Semantic Memory Retrieval**: Transforms raw photo/video filenames, exif metadata, and user notes into dense semantic embeddings using local Truncated SVD / LSA. Users can retrieve memories using natural language queries (e.g. *"quiet rainy night with tea"* or *"festival smiling with friends"*) without uploading private family photos to external cloud APIs.
2. **Local Creative Motion Previews**: Automatically scans photo sequences, computes Farneback optical flow manifolds, and identifies coherent image clusters suitable for automated video montage / reel creation.

---

## 2. Architecture & Design Principles

```
  +-------------------------------------------------------+
  |              UGREEN AI NAS (UGOS System)              |
  +-------------------------------------------------------+
                             |
             [Private Storage Volumes: /media/photos]
                             |
                             v
             +-------------------------------+
             |   exif & metadata extractor   |
             +-------------------------------+
                             |
                             v
             +-------------------------------+
             | Dense Semantic Vector Engine  |
             |   (Local TF-IDF + SVD / LSA)  |
             +-------------------------------+
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
  [Natural Language Search]             [Temporal Motion Clustering]
  Query: "celebration with family"      Farneback Optical Flow Score
  Matches via Cosine Similarity         Filters Best Keyframes for Reels
```

- **Zero Cloud Leakage**: All computation occurs locally on the NAS CPU/NPU.
- **Ultra-Lightweight Footprint**: Memory usage strictly <150MB RAM, allowing concurrent execution alongside other UGOS system services.
- **Instant Cold Start**: Indexing latency <0.2s for 1,000 memory items.

---

## 3. Skill Capabilities & Actions

### Action 1: `index_vault(directory)`
Scans a target directory on the NAS, extracts text tags, metadata, and user descriptions, and compiles a dense semantic vector index saved locally as `vault_vectors.npz`.

### Action 2: `search_memories(query, top_k=5)`
Computes the query vector, calculates cosine similarity across the indexed photo corpus, and returns the top matching memories with similarity scores and direct file paths.

### Action 3: `evaluate_motion_cluster(image_list)`
Analyzes a cluster of related photos, computes inter-frame motion vector variance, and verifies whether the group is suitable for rendering an automated temporal story.

---

## 4. Requirements & Compatibility
- **OS**: UGREEN UGOS (Debian/Linux based).
- **Runtime**: Python 3.9+ with standard library, numpy, and scipy/sklearn (pre-installed or bundled in virtual environment).
- **Storage**: <10 MB skill package footprint.
