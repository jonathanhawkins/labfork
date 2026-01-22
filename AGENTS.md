# Repository Guidelines

## Project Structure & Module Organization
This repo is organized by pipeline stage. Key locations:
- `backend/`: FastAPI API + prosody analysis (`main.py`, `prosody_analyzer.py`).
- `frontend/`: Next.js UI (`app/`, `components/`, Tailwind config).
- `training/`: dataset prep and training scripts (`prepare_dataset.py`, `train_deepseek.py`, `config/`).
- `inference/`: speech generation (`generate.py`).
- `data/`: audio datasets (raw/processed/labeled/splits).
- `models/`: checkpoints and downloaded models.
- `scripts/`: setup and model download utilities.
- `docs/`: design/tech notes; `PRD.md` at repo root for architecture.

## Build, Test, and Development Commands
Common workflows:
```bash
# setup
./scripts/setup_mac.sh
./scripts/setup_linux.sh

# download models
python scripts/download_models.py

# backend API
cd backend && source venv/bin/activate && python main.py

# frontend UI
cd frontend && npm run dev

# build/lint frontend
cd frontend && npm run build
cd frontend && npm run lint

# training
cd training
python prepare_dataset.py --input ../data/labeled --output ../data/splits
python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard

# inference
cd inference
python generate.py --model ../models/checkpoints/voice_v1/best.pt --text "Hello" --output out.wav
```

## Coding Style & Naming Conventions
- Python: 4-space indentation, snake_case for functions/vars, keep modules focused.
- TypeScript/TSX: 2-space indentation, double quotes, PascalCase React components.
- Follow existing patterns in `backend/` and `frontend/app/` before introducing new abstractions.

## Testing Guidelines
- No formal test suite is configured today; smoke test by running the backend + frontend flow.
- If adding tests, use pytest conventions (`backend/tests/test_*.py`) and keep fixtures minimal.

## Commit & Pull Request Guidelines
- Commit messages follow short, imperative summaries (e.g., "Fix frontend null safety...").
- PRs should include: clear description, linked issue (if any), test or smoke steps, and UI screenshots for frontend changes.
- Note any training config changes and expected hardware impact.

## Automation & Safety Notes
- Prefer `npm` for frontend tooling.
- Run `npm run lint` before commits that touch `frontend/`.
- Ask before adding new production dependencies or deleting files.
- Create a git checkpoint before significant changes.
- Configure local endpoints via `NEXT_PUBLIC_API_URL` (frontend) and `DATA_DIR` (backend).
