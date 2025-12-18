# Semantic Climate Phase Space — System Documentation

Technical documentation for the Semantic Climate analysis system.

## Contents

### Core Documentation

| Document | Description |
|----------|-------------|
| [00-OVERVIEW.md](00-OVERVIEW.md) | Architecture, data flow, key concepts |
| [01-CORE-METRICS.md](01-CORE-METRICS.md) | Morgoulis metrics (Δκ, α, ΔH) with fixes |
| [02-EXTENSIONS.md](02-EXTENSIONS.md) | Ψ vector, attractor basins, trajectory dynamics |
| [03-WEB-APP.md](03-WEB-APP.md) | FastAPI backend, services, WebSocket protocol |
| [04-BATCH-ANALYSIS.md](04-BATCH-ANALYSIS.md) | Archive analysis pipeline, output format |
| [05-API-REFERENCE.md](05-API-REFERENCE.md) | Class and function reference |

### Appendices

| Document | Description |
|----------|-------------|
| [A1-METRICS-MATHEMATICAL-REVIEW.md](A1-METRICS-MATHEMATICAL-REVIEW.md) | Full mathematical analysis of original metric problems |
| [A2-METRIC-FIX-VALIDATION.md](A2-METRIC-FIX-VALIDATION.md) | Validation test cases and results for all fixes |

## Quick Links

### For Users

- **Running the web app:** [03-WEB-APP.md#running-the-app](03-WEB-APP.md#running-the-app)
- **Batch analysis:** [04-BATCH-ANALYSIS.md#usage](04-BATCH-ANALYSIS.md#usage)
- **Interpreting results:** [04-BATCH-ANALYSIS.md#interpreting-results](04-BATCH-ANALYSIS.md#interpreting-results)

### For Developers

- **Architecture overview:** [00-OVERVIEW.md](00-OVERVIEW.md)
- **API reference:** [05-API-REFERENCE.md](05-API-REFERENCE.md)
- **Core metrics math:** [01-CORE-METRICS.md](01-CORE-METRICS.md)

### Key Concepts

- **Attractor basins:** [02-EXTENSIONS.md#attractor-basins](02-EXTENSIONS.md#attractor-basins)
- **Coupling modes:** [00-OVERVIEW.md#coupling-modes-trajectory-aware](00-OVERVIEW.md#coupling-modes-trajectory-aware)
- **Epistemic risk:** [00-OVERVIEW.md#epistemic-risk-assessment](00-OVERVIEW.md#epistemic-risk-assessment)

## License

- Core metrics (Δκ, α, ΔH): MIT (Morgoulis 2025)
- Extensions & application: ESL-A (Earthian Stewardship License)
