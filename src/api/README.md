# DeepTutor Mock API

A comprehensive mock API implementation for the DeepTutor AI education system, providing all the endpoints needed by the web frontend.

## Overview

This mock API provides realistic endpoints and data for development and testing purposes. It includes WebSocket support for real-time features and comprehensive CRUD operations for all system components.

## Features

### 🧠 Knowledge Base Management (`/api/v1/knowledge`)
- List, create, update, and delete knowledge bases
- Document upload with progress tracking
- Knowledge base statistics and metadata
- Document management within knowledge bases

### 🔍 Problem Solver (`/api/v1/solve`)
- WebSocket endpoint for real-time problem solving
- Streaming responses with progress updates
- Multiple knowledge base support
- Solution history and analytics

### ❓ Question Generation (`/api/v1/question`)
- WebSocket-based question generation
- Multiple question types (multiple choice, short answer, essay)
- Difficulty level adjustment
- Question mimicking from uploaded examples
- Template-based generation

### 🔬 Research Tools (`/api/v1/research`)
- WebSocket research sessions
- Multiple research tools simulation
- Comprehensive report generation
- Research history and analytics
- Configurable research modes

### 📚 Notebook Management (`/api/v1/notebook`)
- CRUD operations for notebooks and records
- Record import/export functionality
- Search and filtering capabilities
- Markdown export support
- Cross-notebook record management

### ✍️ Co-Writer (`/api/v1/cowriter`)
- AI-assisted writing sessions
- Text improvement suggestions
- Auto-annotation features
- TTS (Text-to-Speech) integration
- Document export in multiple formats

### 🎓 Guided Learning (`/api/v1/guide`)
- Personalized learning path generation
- Interactive Q&A sessions via WebSocket
- Knowledge point progression
- Learning session management
- Progress tracking

### 💡 Idea Generation (`/api/v1/ideagen`)
- WebSocket-based idea generation
- Research idea templates
- Customizable difficulty levels
- Idea categorization and saving
- Integration with notebook system

### ⚙️ Settings & System Status (`/api/v1/settings`)
- System configuration management
- LLM provider configuration
- Environment variable management
- Health monitoring and testing
- Backup management

## API Structure

```
src/api/
├── __init__.py              # Module initialization
├── main.py                  # FastAPI application entry point
├── requirements.txt         # Python dependencies
├── knowledge.py             # Knowledge base endpoints
├── solver.py                # Problem solving endpoints
├── question.py              # Question generation endpoints
├── research.py              # Research tool endpoints
├── notebook.py              # Notebook management endpoints
├── cowriter.py              # Co-writer functionality endpoints
├── guide.py                 # Guided learning endpoints
├── ideagen.py               # Idea generation endpoints
└── settings.py              # Settings and system status endpoints
```

## Quick Start

### Installation

```bash
cd src/api
pip install -r requirements.txt
```

### Running the API

```bash
# Development mode with auto-reload
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Or run directly
python main.py
```

### API Documentation

Once running, access the interactive API documentation:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Endpoint Categories

### REST Endpoints
Most endpoints follow RESTful conventions:
- `GET` - Retrieve data
- `POST` - Create new resources
- `PUT` - Update existing resources
- `DELETE` - Remove resources

### WebSocket Endpoints
Real-time features use WebSocket connections:
- `/api/v1/solve/ws` - Problem solving
- `/api/v1/question/ws` - Question generation
- `/api/v1/research/ws` - Research sessions
- `/api/v1/guide/session/{session_id}/chat` - Learning Q&A
- `/api/v1/ideagen/ws` - Idea generation

## Mock Data Features

### Realistic Data Generation
- Dynamic content based on request parameters
- Randomized but consistent responses
- Proper error handling and validation
- Simulated processing delays for realism

### Progress Tracking
- WebSocket progress updates
- Status indicators for long-running operations
- Completion notifications with statistics

### Data Persistence
- In-memory storage for session duration
- Consistent data across related endpoints
- Proper relationship management

## Configuration

### CORS Settings
The API is configured to allow cross-origin requests from any domain for development purposes. In production, update the CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables
The API can be configured using environment variables:
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8001)
- `DEBUG` - Enable debug mode (default: True)

## Testing

### Manual Testing
Use the built-in Swagger UI at `/docs` for interactive testing of all endpoints.

### Automated Testing
```bash
pytest tests/
```

## Integration with Frontend

This mock API is designed to work seamlessly with the Next.js frontend. The API base URL should be configured in the frontend's environment variables:

```bash
# Frontend .env.local
NEXT_PUBLIC_API_BASE=http://localhost:8001
NEXT_PUBLIC_WS_BASE=ws://localhost:8001
```

## Mock Data Highlights

### Knowledge Bases
- Pre-populated with AI textbook, Python docs, and research papers
- Realistic document counts and sizes
- Processing status simulation

### Problem Solutions
- Context-aware responses based on question keywords
- Mathematical formulas using LaTeX
- Code examples with syntax highlighting
- Comprehensive explanations

### Research Reports
- Domain-specific content for AI and climate change
- Professional report structure
- Citations and references
- Executive summaries

### Learning Content
- Structured knowledge points
- Progressive difficulty levels
- Interactive Q&A responses
- Personalized learning paths

## Development Notes

### Adding New Endpoints
1. Create endpoint functions in the appropriate module
2. Add proper type hints and documentation
3. Include error handling
4. Update the main router in `main.py`

### WebSocket Implementation
WebSocket endpoints use connection managers for proper connection handling and message broadcasting.

### Mock Data Strategy
Mock data is generated dynamically but consistently, providing realistic responses while maintaining data relationships across endpoints.

## Production Considerations

This is a mock API for development and testing. For production deployment:

1. Replace mock data with real database connections
2. Implement proper authentication and authorization
3. Add rate limiting and security middleware
4. Configure proper CORS origins
5. Add comprehensive logging and monitoring
6. Implement data validation and sanitization

## Support

For questions or issues with the mock API, refer to the main project documentation or create an issue in the project repository.