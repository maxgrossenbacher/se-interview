# LangGraph Agent with Web Search

A simple LangGraph agent that can search the web using DuckDuckGo, exposed via a FastAPI server.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- OpenAI API key

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Create a `.env` file from the example:

```bash
cp .env.example .env
```

3. Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_actual_api_key
```

## Running the API

Start the FastAPI server:

```bash
poetry run uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /chat

Send a message to the agent and receive a response.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about AI?"}'
```

**Response:**

```json
{
  "response": "Based on my search, here are the latest developments in AI..."
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```


## Project Structure

```
se-interview/
├── pyproject.toml   # Poetry dependencies
├── .env.example     # Environment variable template
├── README.md        # This file
├── agent.py         # LangGraph agent implementation
└── api.py           # FastAPI server
```

## How It Works

1. The agent receives a user message via the `/chat` endpoint
2. It calls GPT-4o with the message and available tools (DuckDuckGo search)
3. If the LLM decides to search, it executes the search and feeds results back
4. The loop continues until the LLM provides a final response
5. The response is returned to the user


## Design Decisions

### Tool Architecture
Implemented 3 specialized tools rather than a single general-purpose search tool:

1. **`find_flight_options`** — searches for flights between origin and destination cities for given dates
2. **`find_hotel_options`** — searches for hotel options in the destination city for given dates  
3. **`find_nearby_attractions`** — searches for popular attractions, activities, and events in the destination city

### Why Separate Tools?
Each tool has its own Pydantic input schema (`FlightDetails`, `HotelDetails`, `AttractionDetails`) which serves two purposes:
- **Structured parameter extraction** — GPT-4o reads the schema to extract typed fields (origin, destination, dates) directly from natural language rather than passing a raw query string
- **Separation of concerns** — each tool focuses on a single task, making it easier to instrument, test, and debug independently in Phoenix traces

### Search Backend
DuckDuckGo was chosen as the search backend because it requires no API key and works out of the box. In production, these tools would be replaced with dedicated APIs:
- Flights → Google Flights (SerpAPI)
- Hotels → Booking.com or Expedia API
- Attractions → Google Places API or local events calender or sites like tripadvisor

### Observability
Separating tools by domain means Phoenix traces show exactly which tool was invoked for each query — making it straightforward to evaluate tool usage correctness and debug incorrect tool selection.

## Evaluation Methodology

### Running Evaluations

Ensure Phoenix is running and spans exist:

```bash
docker compose up -d phoenix
python evaluate.py
```

### 1. User Frustration

Uses Phoenix's built-in `USER_FRUSTRATION_PROMPT_TEMPLATE` with `llm_classify`.
An LLM judge (GPT-4o) reads each `User: ... / Assistant: ...` conversation and classifies
it as **"frustrated"** or **"ok"**.

- **Why this metric:** Detects when the agent fails to help, misunderstands, or gives unhelpful responses.
- **Labels:** `frustrated` | `ok`
- **Results logged to:** Phoenix span annotations + `eval_frustration.csv`
- **Frustrated interactions** are filtered and saved as a Phoenix dataset for regression testing.

### 2. Tool Selection Correctness

A custom `ClassificationTemplate` judges whether the agent picked the right
tool(s) for the user's query given the available tools:
`find_flight_options`, `find_hotel_options`, `find_nearby_attractions`, `duckduckgo_search`.

- **Why this metric:** The agent's value depends on routing queries to specialized tools rather than falling back to generic search.
- **Labels:**
  - `correct` (1.0) — exactly the right tool(s), or correctly deferred (e.g., asked clarifying questions).
  - `acceptable` (0.5) — reasonable but sub-optimal choice (e.g., generic search instead of specialized tool).
  - `incorrect` (0.0) — wrong tool, missing tool call, or tool call when clarification was needed.
- **Results logged to:** Phoenix span annotations + `eval_tool_selection.csv`