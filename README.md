# Tazotron Inference

Inference service compatible with the current backend contract.

## Run Locally

```bash
uv sync --group dev
uv run tazotron-inference
```

For development with auto-reload:

```bash
uv run uvicorn tazotron.inference.api:app --reload
```

## Test

```bash
uv run pytest
```

The HTTP e2e tests use `fastapi.testclient.TestClient` and mock the model layer, so they do not load a real checkpoint.

## Endpoints

- `GET /health`
- `GET /models`
- `GET /models/{externalId}/metrics`
- `POST /classify`

## Runtime Model Source

The service loads one model lazily from ClearML using the configured project, task, and artifact alias. The artifact is expected to be the best checkpoint produced by `finetune_radiologynet_binary_necrosis.ipynb`.
