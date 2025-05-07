# MCP Server

MCP (Model Context Protocol) Server is a modular AI assistant engine that runs on your machine to extend what a language model can do beyond just generating text.

## Features

- 🧠 AI Router: Switch between online APIs (OpenRouter, OpenAI) and local models (via Ollama)
- 🛠️ Tool Plugins: Poetry generation, code assistance, Mac cleanup, daily agent, and more
- 📚 Knowledge & Memory: Vector database storage for personal data and conversations
- 🎙️ Multimodal: Speech-to-text, text-to-speech, and image processing capabilities
- 📶 API Clients: OpenRouter.ai, HuggingFace, Google Search, WolframAlpha

## Use Cases

- ✍️ **Poetry Mode**: Generate, complete, or enhance verses based on your own poetry database
- 🧑‍💻 **Coding Mode**: Code generation, debugging, snippet search, and more
- 🎯 **Life Assistant Mode**: Daily briefings, schedule tracking, and task management
- 📚 **Education & Research**: Topic explanations, research article summarization
- 🧠 **Custom Persona**: Create a chatbot that reflects your unique style and preferences

## Installation

1. Clone this repository:
```bash
git clone https://github.com/darshan-regmi/mcp-server.git
cd mcp-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on the example:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Start the server:
```bash
python -m app.main
```

5. Open your browser and navigate to:
```
http://localhost:8000
```

## API Keys

MCP Server can use various AI models and services. You'll need to obtain API keys for the services you want to use:

- **OpenAI API**: https://platform.openai.com/
- **OpenRouter**: https://openrouter.ai/
- **HuggingFace**: https://huggingface.co/
- **Google Custom Search**: https://developers.google.com/custom-search/
- **Wolfram Alpha**: https://developer.wolframalpha.com/
- **ElevenLabs**: https://elevenlabs.io/

For offline usage, you can use [Ollama](https://ollama.ai/) to run models locally.

## Project Structure

```
mcp-server/
├── app/                  # Main application code
│   ├── routers/          # API routers
│   ├── static/           # Web interface
│   └── main.py           # Application entry point
├── tools/                # Tool plugins
├── memory/               # Vector database storage
├── config/               # Configuration files
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables (not in repo)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Additional Setup

1. Install Ollama for local model support (optional):
   ```bash
   # Visit https://ollama.ai/ for installation instructions
h   ```

## Usage

### Starting the Server

```bash
python -m app.main
```

You can configure the server using environment variables in your `.env` file:
- `MCP_HOST`: Host address to bind to (default: 0.0.0.0)
- `MCP_PORT`: Port to listen on (default: 8000)
- `MCP_MEMORY_DIR`: Directory for storing memory data (default: memory)

The server will be available at http://localhost:8000 by default.

### API Endpoints

#### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a poem about dreams", "mode": "poetry"}'
```

#### Tools

```bash
curl -X POST http://localhost:8000/tools/execute \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "poetry_gen", "parameters": {"topic": "dreams", "style": "sonnet"}}'
```

#### Memory

```bash
curl -X POST http://localhost:8000/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What did I write about dreams?", "collection": "notes"}'
```

## Architecture

```
📦 MCP Server (FastAPI + Python)
├── 🧠 AI Router (API vs Local LLM via Ollama)
├── 🛠️ Tool Plugins
│   ├── poetry_gen.py
│   ├── code_assist.py
│   ├── mac_cleanup.py
│   ├── daily_agent.py
│   └── memory_manager.py
├── 📚 Knowledge & Memory
│   ├── Local Notes (MD, TXT, Notion export)
│   ├── Vector Store (ChromaDB / Qdrant)
│   └── Browser Tools / Docs Reader
├── 🎙️ Multimodal
│   ├── Speech-to-text (Whisper)
│   ├── TTS (Piper / ElevenLabs)
│   └── Image Tools
└── 📶 API Clients
    ├── OpenRouter.ai
    ├── HuggingFace
    ├── Google Search
    └── WolframAlpha
```

### API Response Format

```json
{
  "message": "Here's a sonnet about dreams...",
  "tool_calls": [
    {
      "name": "poetry_gen",
      "parameters": {"topic": "dreams", "style": "sonnet"}
    }
  ],
  "sources": [
    {"title": "My Dream Journal", "content": "Excerpt from your notes..."}
  ],
  "mode": "poetry",
  "model_used": "claude-3-opus-20240229",
  "processing_time": 1.25
}
```

## Privacy and Security

- 🔒 Your data stays local - no one sees your personal information
- 🔑 API keys are stored securely in your `.env` file
- 🛡️ Offline-capable with local models via Ollama
- 🌐 Consider using SSL/TLS for secure communication in production
- 🔥 Restrict access to the MCP server port using a firewall

## Extending MCP Server

You can extend the MCP server by adding your own tool plugins:

1. Create a new Python file in the `tools` directory
2. Define a class that inherits from the `Tool` base class
3. Implement the `execute` method
4. Your tool will be automatically loaded when the server starts

## License

This project is open source and available under the MIT License.
