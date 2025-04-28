# Electronic AI Assistant

An intelligent AI assistant built with modern AI/ML technologies.

## Project Structure

```
/data       --> Raw and processed datasets
/models     --> Trained models and checkpoints
/src        --> Core source code (models, agents, utilities)
/configs    --> Config files for models, agents, experiments
/tests      --> Unit and integration tests
/agents     --> Tools, memory modules, and agentic components
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your environment variables:
```
OPENAI_API_KEY=your_api_key_here
```

## Development

- Run tests: `pytest`
- Format code: `black . && isort .`
- Type checking: `mypy .`

## Documentation

The documentation is built using MkDocs. To view the documentation locally:

```bash
mkdocs serve
```

Then visit `http://localhost:8000`

## License

MIT License 