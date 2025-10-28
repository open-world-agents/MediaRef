# Contributing to MediaRef

Thank you for your interest in contributing to MediaRef!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/open-world-agents/mediaref.git
cd mediaref
```

2. Install in development mode with all dependencies:
```bash
pip install -e ".[loader,dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mediaref --cov-report=html

# Run specific test
pytest tests/test_mediaref.py::TestMediaRefLoading::test_to_rgb_array
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Keep functions focused and small

## Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Reporting Issues

Please include:
- Python version
- MediaRef version
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Open an issue or discussion on GitHub.

