# Batch Analysis Pipeline

**Script:** `scripts/batch_analyze_conversations.py`
**License:** ESL-A

## Overview

The batch analyzer processes archived conversation exports (e.g., ChatGPT exports) to compute semantic climate metrics, detect attractor basins, and assess epistemic risk across dialogue history.

## Input Format

### Expected JSON Structure

```json
{
    "id": "conversation-uuid",
    "title": "Conversation Title",
    "conversation_start_iso": "2023-07-29",
    "turns": [
        {
            "role": "user",
            "content": "User's message text",
            "timestamp": "2023-07-29T10:30:00Z"
        },
        {
            "role": "assistant",
            "content": "AI's response text",
            "timestamp": "2023-07-29T10:30:15Z"
        }
    ]
}
```

**Notes:**
- `role` must be "user", "assistant", or "system" (system messages are filtered out)
- Empty content is skipped
- Timestamps are optional but preserved in output

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: JSON Files                            │
│    /path/to/conversations/*.json                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Load & Extract                               │
│    load_conversation() → extract_turns()                        │
│    Filter system messages, extract speaker labels               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generate Embeddings                          │
│    EmbeddingService.embed_batch(texts)                          │
│    sentence-transformers or Ollama                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Windowed Analysis                            │
│    for window in sliding_windows:                               │
│        MetricsService.analyze(                                  │
│            embeddings, texts, speakers,                         │
│            trajectory_metrics, previous_metrics                 │
│        )                                                        │
│        Accumulate trajectory_metrics                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final Analysis                               │
│    Full-conversation metrics with complete context              │
│    Dominant mode, trajectory direction, risk assessment         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output: Analysis JSON                        │
│    /path/to/analysis/{filename}_analysis.json                   │
│    batch_summary.json                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Command Line

```bash
# Single conversation
python scripts/batch_analyze_conversations.py /path/to/conversation.json

# Directory of conversations
python scripts/batch_analyze_conversations.py /path/to/conversations/

# With options
python scripts/batch_analyze_conversations.py /path/to/conversations/ \
    --output /path/to/output/ \
    --min-turns 10 \
    --embedding-model all-mpnet-base-v2 \
    --embedding-backend sentence-transformers
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `input_path/../analysis/` | Output directory |
| `--min-turns, -m` | 10 | Minimum turns for analysis |
| `--embedding-model` | `all-mpnet-base-v2` | Embedding model name |
| `--embedding-backend` | `sentence-transformers` | `sentence-transformers` or `ollama` |

---

## Analysis Pipeline Detail

### 1. Turn Extraction

```python
def extract_turns(conversation: dict) -> list[dict]:
    """Filter system messages, normalize speaker labels."""
    # Returns: [{'speaker': 'human'|'ai', 'text': str, 'timestamp': str}]

def extract_turn_data(turns: list[dict]) -> tuple[list[str], list[str]]:
    """Extract parallel lists of texts and speakers."""
    # Returns: (texts, speakers)
```

### 2. Windowed Analysis

The conversation is analyzed in overlapping windows to capture trajectory evolution:

```python
window_size = min_turns  # Default: 10
step_size = max(1, (len(turns) - window_size) // 10)  # ~10 snapshots

accumulated_trajectory_metrics = []

for i in range(0, len(turns) - window_size + 1, step_size):
    window_embeddings = embeddings[i:i+window_size]
    window_texts = texts[i:i+window_size]
    window_speakers = speakers[i:i+window_size]

    result = metrics_service.analyze(
        embeddings=window_embeddings,
        turn_texts=window_texts,
        turn_speakers=window_speakers,
        previous_metrics=previous_metrics,
        semantic_shifts=compute_shifts(window_embeddings),
        trajectory_metrics=accumulated_trajectory_metrics
    )

    accumulated_trajectory_metrics.append(result['metrics'])
```

### 3. Semantic Shift Computation

```python
def compute_semantic_shifts(embeddings):
    """Compute cosine distance between consecutive turns."""
    shifts = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        shifts.append(1.0 - sim)  # Distance = 1 - similarity
    return shifts
```

### 4. Final Analysis

Full-conversation analysis with complete context:

```python
final_result = metrics_service.analyze(
    embeddings=all_embeddings,
    turn_texts=texts,
    turn_speakers=speakers,
    previous_metrics=previous_metrics,
    semantic_shifts=all_semantic_shifts,
    trajectory_metrics=accumulated_trajectory_metrics
)
```

---

## Output Format

### Individual Analysis (`{filename}_analysis.json`)

```json
{
    "status": "analyzed",
    "metadata": {
        "id": "conversation-uuid",
        "title": "Conversation Title",
        "start_date": "2023-07-29",
        "turn_count": 23,
        "analyzed_at": "2025-12-11T22:21:01.231888"
    },
    "summary": {
        "dominant_mode": "Exploratory",
        "mode_distribution": {"Exploratory": 12},
        "final_mode": "Exploratory",
        "final_coupling": "Exploratory",
        "epistemic_risk": "low",
        "risk_factors": [],
        "trajectory_direction": {
            "delta_kappa_change": 0.0673,
            "delta_h_change": 0.1311,
            "alpha_change": 0.0,
            "overall": "warming"
        }
    },
    "final_metrics": {
        "delta_kappa": 1.22,
        "alpha": 0.5,
        "delta_h": 0.86,
        "confidence_intervals": {...},
        "trends": {...}
    },
    "final_psi": {
        "composite": 0.358,
        "vector": {
            "semantic": 0.999,
            "temporal": 0.0,
            "affective": -0.71,
            "biosignal": null
        }
    },
    "final_attractor": {
        "name": "Cognitive Mimicry",
        "confidence": 0.90
    },
    "final_coherence": {
        "autocorrelation": 0.067,
        "variance": 0.044,
        "pattern": "transitional",
        "coherence_score": 0.57
    },
    "trajectory": [
        {
            "window_start": 0,
            "window_end": 10,
            "turn_range": "1-10",
            "metrics": {...},
            "mode": "Exploratory",
            "coupling_mode": {...},
            "psi_vector": {...},
            "attractor_basin": {"name": "Cognitive Mimicry", "confidence": 0.5},
            "coherence": {"pattern": "breathing", ...}
        },
        // ... more windows
    ],
    "source_file": "2023-07-29__conversation__23turns.json"
}
```

### Batch Summary (`batch_summary.json`)

```json
{
    "generated_at": "2025-12-11T22:25:00.000000",
    "input_path": "/path/to/conversations",
    "total_files": 10,
    "analyzed": 8,
    "skipped": 1,
    "errors": 1,
    "mode_distribution": {
        "Exploratory": 4,
        "Resonant": 2,
        "Generative": 1,
        "Sycophantic": 1
    },
    "risk_distribution": {
        "low": 6,
        "moderate": 1,
        "high": 1
    },
    "trajectory_distribution": {
        "warming": 3,
        "stable": 3,
        "cooling": 2
    },
    "results": [
        // Full results array for programmatic access
    ]
}
```

---

## Interpreting Results

### Trajectory Evolution

The `trajectory` array shows how the conversation evolved:

```python
# Look for basin shifts
for i, window in enumerate(result['trajectory']):
    print(f"Window {window['turn_range']}: "
          f"{window['attractor_basin']['name']} "
          f"({window['coherence']['pattern']})")
```

Example output:
```
Window 1-10: Cognitive Mimicry (breathing)
Window 2-11: Cognitive Mimicry (breathing)
Window 3-12: Cognitive Mimicry (breathing)
...
Window 12-21: Collaborative Inquiry (locked)
```

### Risk Indicators

```python
if result['summary']['epistemic_risk'] in ['high', 'critical']:
    print(f"Warning: {result['summary']['risk_factors']}")
```

Common risk factors:
- `regressive_sycophancy` — Moving toward compliance
- `epistemic_enclosure_risk` — Closing off exploration
- `complexity_collapse` — Falling curvature
- `semantic_fragmentation` — High entropy, low coherence
- `repetitive_pattern` — Locked coherence pattern

### Basin Transitions

A shift from Cognitive Mimicry to Collaborative Inquiry indicates:
- Increased hedging (genuine uncertainty emerging)
- More balanced turn lengths
- Oscillating Δκ (responsive exploration)

---

## Performance Considerations

### Embedding Generation

The slowest step is embedding generation:
- ~100ms per turn with sentence-transformers (GPU)
- ~500ms per turn with Ollama (CPU)

For large archives:
```bash
# Use sentence-transformers for speed
--embedding-backend sentence-transformers

# Or pre-generate embeddings and cache
```

### Memory Usage

Each embedding is 768 floats × 4 bytes = ~3KB. A 100-turn conversation:
- Embeddings: ~300KB
- Full analysis result: ~50KB JSON

For very large archives (1000+ conversations), process in batches.

---

## Example Workflow

```bash
# 1. Prepare conversations (from ChatGPT export)
python scripts/preprocess_chatgpt_export.py conversations.json

# 2. Run batch analysis
cd semantic-climate-phase-space
python scripts/batch_analyze_conversations.py \
    /path/to/conversations-for-SC/ \
    --output /path/to/analysis/ \
    --min-turns 10

# 3. Review summary
cat /path/to/analysis/batch_summary.json | jq '.mode_distribution'

# 4. Investigate high-risk conversations
cat /path/to/analysis/batch_summary.json | \
    jq '.results[] | select(.summary.epistemic_risk == "high")'
```
