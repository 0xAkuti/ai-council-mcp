# AI Council MCP Server

> **Herramienta de Consenso Multi-IA**: Consulta múltiples modelos de IA en paralelo, sintetiza las respuestas para obtener una mayor precisión y reduce el sesgo de la IA mediante la toma de decisiones en conjunto.

AI Council es un potente servidor MCP (Model Context Protocol) que aprovecha la "sabiduría de las masas" consultando múltiples modelos de IA simultáneamente. Obtén respuestas más fiables y exhaustivas combinando perspectivas de OpenAI, Claude, Gemini y cualquier API compatible con OpenAI.

## ✨ ¿Qué es AI Council?

AI Council transforma la forma en que interactúas con la IA mediante:

- **🔄 Procesamiento en Paralelo**: Consulta múltiples modelos de IA simultáneamente (no secuencialmente).
- **🎯 Reducción de Sesgos**: Utiliza nombres en clave anónimos para evitar el sesgo de síntesis.
- **⚡ Síntesis Inteligente**: Un modelo sintetiza todas las respuestas en una respuesta exhaustiva.
- **🔧 Compatibilidad Universal**: Funciona con OpenAI, OpenRouter y cualquier API compatible con OpenAI.
- **🛡️ Gestión Robusta de Errores**: Degradación gradual cuando los modelos individuales fallan.

**Ideal para**: Preguntas de investigación, análisis complejos, proyectos creativos, decisiones técnicas y cualquier tarea donde múltiples perspectivas de IA aporten valor.

## 📋 Requisitos
- Python 3.10+
- **uv** instalado ([guía de instalación](https://docs.astral.sh/uv/getting-started/installation))
- alternativamente:
  - **pipx** instalado ([guía de instalación](https://pipx.pypa.io/stable/installation)), actualiza la configuración:
    ```json
      "command": "pipx",
      "args": ["run", "ai-council"]
    ```
  - o instalación manual con `pip install ai-council`, y actualiza la configuración:
      ```json
      "command": "ai-council",
      "args": []
    ```

## 🚀 Inicio Rápido

1. Obtén tu clave de API de [OpenRouter](https://openrouter.ai/)

### Configuración en Cursor IDE

2. Abre Cursor Settings → MCP
3. Añade un nuevo servidor MCP y establece tu clave de API:

```json
{
  "ai-council": {
    "command": "uvx",
    "args": ["ai-council"],
    "env": {
      "OPENROUTER_API_KEY": "..."
    }
  }
}
```

### Configuración en Claude Desktop

2. Edita `~/.claude_desktop_config.json` y establece tu clave de API:

```json
{
  "mcpServers": {
    "ai-council": {
      "command": "uvx",
      "args": ["ai-council"],
      "env": {
        "OPENROUTER_API_KEY": "..."
      }
    }
  }
}
```

**¡Eso es todo!** Haz cualquier pregunta compleja y la herramienta AI Council involucrará automáticamente a múltiples modelos. 

Por defecto, utilizará OpenRouter con Claude Sonnet 4, Gemini 2.5 Pro y DeepSeek V3.

### Argumentos de CLI

Usa argumentos de línea de comandos para una configuración rápida; añade cualquiera de estos a los `args` en tu configuración de MCP:

**Argumentos de CLI disponibles**:
- `--openai-api-key`: Tu clave de API de OpenAI.
- `--openrouter-api-key`: Tu clave de API de OpenRouter.  
- `--max-models`: Máximo de modelos a consultar (por defecto: 3).
- `--parallel-timeout`: Tiempo de espera en segundos (por defecto: 60).
- `--log-level`: Nivel de registro (DEBUG, INFO, WARNING, ERROR).
- `--config`: Ruta al archivo de configuración personalizado.

## ⚙️ Configuración Avanzada

Para configuraciones avanzadas, crea un archivo `config.yaml` y enlázalo con `--config ruta/al/config.yaml`:

```yaml
# config.yaml
openai_api_key: "tu_clave_openai_aquí"
openrouter_api_key: "tu_clave_openrouter_aquí"
max_models: 3
parallel_timeout: 90 # en segundos
synthesis_model_selection: "random"  # o "first"

models:
  # use OpenAI API
  - name: "GPT-4o"
    provider: "openai" 
    model_id: "gpt-4o"
    enabled: true # opcional, por defecto true
    
  # use OpenRouter API
  - name: "Claude Sonnet"
    provider: "openrouter"
    model_id: "anthropic/claude-3.5-sonnet"
    code_name: "Bob" # opcional, se asigna automáticamente de lo contrario
  
  # o cualquier API personalizada compatible con OpenAI
  - name: "Perplexity"
    provider: "custom"
    model_id: "llama-3.1-sonar-large-128k-online"
    base_url: "https://api.perplexity.ai"
    api_key: "tu_clave_perplexity_aquí"
    
  # LLM Local (Ollama)
  - name: "Local Llama"
    provider: "custom" 
    model_id: "llama-3b"
    base_url: "http://localhost:11434"
    api_key: "clave-si-es-necesaria"
```

## 📖 Cómo Funciona

AI Council utiliza un sofisticado enfoque de tres fases:

### 1. **Consulta en Paralelo** 
- Consulta simultáneamente los modelos de IA configurados.
- Mantiene el mismo contexto y pregunta para cada modelo.
- Gestiona los fallos con elegancia (continúa con las respuestas exitosas).

### 2. **Análisis Anónimo**
- Asigna nombres en clave (Alpha, Beta, Gamma, etc.) a la respuesta de cada modelo.
- Evita el sesgo de síntesis hacia marcas o proveedores específicos.
- Preserva la calidad de la respuesta eliminando la identidad del modelo.

### 3. **Síntesis Inteligente**
- Selecciona aleatoriamente un modelo para que actúe como sintetizador.
- El sintetizador analiza todas las respuestas anónimas.
- Produce una respuesta exhaustiva combinando las mejores ideas.

## 🤝 Agradecimientos

Este proyecto fue inspirado por [Cognition Wheel](https://github.com/Hormold/cognition-wheel).

AI Council amplía estas ideas con:
- Mayor flexibilidad de configuración.  
- Soporte de OpenRouter para múltiples opciones de modelos con una sola clave de API.
- Soporte para endpoints de API personalizados.
- Gestión de errores y registro mejorados.
- Implementación en Python.
