# Batch Conversation Analysis

Analyze ChatGPT conversation exports for Semantic Climate patterns.

## Usage

```bash
cd semantic-climate-phase-space

# Analyze a single conversation
TOKENIZERS_PARALLELISM=false python3 scripts/batch_analyze_conversations.py \
  /path/to/conversation.json

# Analyze a directory of conversations
TOKENIZERS_PARALLELISM=false python3 scripts/batch_analyze_conversations.py \
  /path/to/conversations/ --min-turns 10
```

## Input Format

Preprocessed JSON files with this structure:

```json
{
  "id": "uuid",
  "title": "Conversation Title",
  "conversation_start_iso": "2023-07-29",
  "turns": [
    {"role": "user", "content": "...", "timestamp": 1690675159},
    {"role": "assistant", "content": "...", "timestamp": 1690675181}
  ]
}
```

## Output

Results saved to `analysis/` subdirectory:

- `*_analysis.json` — Individual conversation analysis with:
  - Metrics trajectory (Δκ, ΔH, α over sliding windows)
  - Coupling mode classifications
  - Coherence patterns
  - Ψ vector and attractor basin detection

- `batch_summary.json` — Aggregate statistics:
  - Mode distribution across conversations
  - Epistemic risk distribution
  - Trajectory direction patterns

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output, -o` | `input_path/../analysis/` | Output directory |
| `--min-turns, -m` | 10 | Minimum turns for analysis |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence-transformer model |

## Notes

- Conversations with fewer than `min-turns` are skipped
- First run downloads the embedding model (~90MB)
- Processing time scales with conversation length and count
