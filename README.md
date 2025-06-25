# AI Council MCP Server

An MCP server that consults multiple AI models in parallel and synthesizes their responses into comprehensive answers using a "wisdom of crowds" approach.

## Features

- **Configurable Models**: Support for both OpenAI and OpenRouter APIs with easy configuration via YAML
- **Parallel Processing**: All models are queried simultaneously for faster results
- **Bias Reduction**: Anonymous code names prevent synthesizer bias toward specific models
- **Flexible Configuration**: Choose which models to use and how many to consult
- **Detailed Logging**: Comprehensive debug logs for transparency and troubleshooting
- **Robust Error Handling**: Graceful degradation when individual models fail

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-council

# Install with uv
uv sync

# Run the server
uv run ai-council
```

### Option 2: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python -m ai_council.main
```

## Configuration

### Environment Variables

Set the following environment variables for the AI services you want to use:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

### Model Configuration

Edit `config.yaml` to configure which models to use:

```yaml
models:
  - name: "gpt-4o"
    provider: "openai"
    model_id: "gpt-4o"
    code_name: "Alpha"
    enabled: true
  - name: "claude-3.5-sonnet"
    provider: "openrouter"
    model_id: "anthropic/claude-3.5-sonnet"
    code_name: "Beta"
    enabled: true
  # Add more models as needed

settings:
  max_models: 3  # How many models to consult simultaneously
  parallel_timeout: 120  # Timeout for parallel calls in seconds
  synthesis_model_selection: "random"  # "random" or "first"
```

## Usage with MCP Clients

### Cursor IDE

1. **Open Cursor Settings**:
   - Go to Settings → MCP
   - Click "Add new MCP server"

2. **Configure the server**:
   ```json
   {
     "ai-council": {
       "command": "uv",
       "args": ["run", "ai-council"],
       "cwd": "/path/to/ai-council",
       "env": {
         "OPENAI_API_KEY": "your_openai_key",
         "OPENROUTER_API_KEY": "your_openrouter_key"
       }
     }
   }
   ```

3. **Test the integration**:
   - Enter Agent mode in Cursor
   - Ask a complex question that would benefit from multiple AI perspectives
   - The `ai_council` tool should be automatically triggered

## How It Works

AI Council follows a three-phase process:

1. **Parallel Consultation**: Simultaneously queries the configured AI models
2. **Anonymous Analysis**: Uses code names (Alpha, Beta, Gamma, etc.) to eliminate bias during synthesis
3. **Smart Synthesis**: Randomly selects one of the models to act as a synthesizer, which analyzes all responses and produces a final, comprehensive answer

## API

The server provides a single tool called `ai_council` with the following parameters:

- `context`: Background information and context for the problem
- `question`: The specific question you want answered
