# Contributing

Contributions are welcome, especially improvements for difficult camera motion,
rolling-shutter artifacts, cloud illumination, and new media formats.

## Development setup

```bash
git clone https://github.com/johncotter3/lightning-frame-stack.git
cd lightning-frame-stack
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
pytest --cov=lightning_frame_stack --cov-report=term-missing
python -m build
```

Please avoid committing private storm footage, location-identifying media, or
large videos. A small synthetic or explicitly redistributable test fixture is
preferred.

## Pull requests

Keep each pull request focused, explain the user-visible behavior, and include a
test for bug fixes or new processing behavior whenever practical.
