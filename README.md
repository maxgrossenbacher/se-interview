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

## Running the Application

### With Docker (recommended)

Start both the agent and Phoenix observability:

```bash
docker compose up -d
```

- Agent API: `http://localhost:8000`
- Phoenix UI: `http://localhost:6006`

## API Endpoints

### POST /chat

Send a message to the agent and receive a response.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Plan me a 5 day trip to tokyo from denver from June 5th to June 10th"}'
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
Each tool has its own input schema (`FlightDetails`, `HotelDetails`, `AttractionDetails`) which serves two purposes:
- **Structured parameter extraction** — GPT-4o reads the schema to extract typed fields (origin, destination, dates) directly from natural language rather than passing a raw query string
- **Separation of concerns** — each tool focuses on a single task, making it easier to instrument, test, and debug independently

### Search Backend
DuckDuckGo was chosen as the search backend because it requires no API key and works out of the box. In production, these tools would be replaced with dedicated APIs:
- Flights → Google Flights (SerpAPI)
- Hotels → Booking.com or Expedia API
- Attractions → Google Places API or local events calender or sites like tripadvisor

### Observability
Separating tools by domain means Phoenix traces show exactly which tool was invoked for each query making it easy to evaluate tool usage correctness and debug incorrect tool selection.

## Evaluation Methodology
The evaluate.py uses LLM-as-a-Judge to evaluate each user interaction and score it based on two evaluators. User frusteration with the agent and whether the correct tool was selected for the user query.

### Evaluation Methods
1. Frusterated - was the user frusterated when chating with the agent
  a. frusterated - the user was frusterated with the agent interaction and likely did not get the response they were looking for
  b. ok - the user was not frusterated as the agent likely provided helpful information relevant to the user query
2. Tool Selection - was the right tool selected for the user query
  a. correct - the right tool or tools were identify by the agent to use to return relevant results for the user query
  b. incorrect - the wrong tool or tools were identify by the agent and return irrelevant results for the user query

### How it works

1. `evaluate.py` fetches all root spans from the `travel-agent` dataset
2. For each span, it extracts the user input and agent output from the span attributes
3. Each conversation is passed through two `llm_classify` calls — one per evaluator — where GPT-4o acts as the judge
4. The frustration evaluator uses Phoenix's built-in `USER_FRUSTRATION_PROMPT_TEMPLATE`
5. The tool selection evaluator uses a custom evaluator defined in `evals/tool_selection.py`
6. Results are logged back to Phoenix as span annotations
7. Results are also exported to `eval_frustration.csv` and `eval_tool_selection.csv`

```bash
# Run evaluations
docker compose up -d
python evaluate.py
```

## Production Architecture

### Architecture Diagram

```text
     ┌────────┐
     │ Client │
     └───┬────┘
         │  POST /chat
         ▼
  ┌──────────────┐
  │ Agent Service│
  │  (FastAPI)   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐         ┌───────────────────────────────────┐
  │   LLM API    │         │         DuckDuckGo Search         │
  │  (OpenAI     │<─tools->|                                   │
  │   GPT-4o)    │         │  ┌─────────┐┌────────┐┌────────┐  │
  └──────────────┘         │  │ Flights ││ Hotels ││Attract.│  │
         |                 │  └─────────┘└────────┘└────────┘  │
         |                 └───────────────────────────────────┘
         │
         │ OTel traces (auto-instrumented)
         ▼
  ┌──────────────┐
  │   Phoenix    │
  │  (traces,    │
  │  evals, LLM  │
  │  observ.)    │
  └──────────────┘
```

**How it flows:**
1. Client sends a message to the Agent Service
2. Agent Service calls GPT-4o with the message and available tools
3. GPT-4o decides which tool(s) to invoke (find_flight_options, find_hotel_options, find_nearby_attractions)
4. Each tool executes a DuckDuckGo search and returns structured results
5. GPT-4o synthesizes the results into a response
6. Every step is auto-traced via OpenTelemetry and sent to Phoenix

**To scale this to production**, add a load balancer in front of multiple stateless Agent Service instances

### Scaling Strategy

- **Stateless agent containers** — each FastAPI instance holds no session state, so horizontal auto-scaling is straightforward. Scale based on request queue depth.
- **Async tool execution** — when the agent calls multiple tools (e.g., flights + hotels + attractions for an itinerary), execute them in parallel rather than sequentially.
- **Queue-based processing** — for complex multi-tool queries, accept the request immediately and process asynchronously. Return results via WebSocket/SSE or a polling endpoint.
- **Rate limiting at the gateway** — protect downstream LLM and search APIs from bursts. Use per-user and global rate limits.

### Latency Optimization

- **Response streaming** — stream LLM output to the client via SSE as tokens are generated, rather than waiting for the full response. Users see output immediately.
- **Redis cache for search results** — flight/hotel/attraction searches for the same query can be cached with a TTL. Avoids redundant API calls.
- **LLM prompt caching** — use provider-level prompt caching (OpenAI cached tokens, Anthropic prompt caching) to reduce latency and cost for repeated system prompts.
- **Connection pooling** — reuse HTTP connections to LLM and search APIs rather than opening new connections per request.
- **Edge deployment** — deploy agent services in multiple regions to reduce round-trip latency for geographically distributed users.

### Cost Considerations

- **LLM token costs** — the largest cost driver. Mitigate with: (1) prompt caching to avoid re-processing the system prompt, (2) smaller models for simple routing decisions vs. full GPT-4o for final responses, (3) token budgets per request.
- **Search API costs** — replace free DuckDuckGo with paid APIs (SerpAPI, Google Places) in production. Cache aggressively to reduce call volume.
- **Eval costs** — LLM-as-judge evaluations cost ~$0.01-0.03 per span. Run evals on sampled golden dataset rather than all historical user requests.