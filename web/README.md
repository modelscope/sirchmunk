# Web Frontend

The Web frontend is a Next.js application that provides the user interface for the OpenCowork system.

## 📋 Overview

The frontend provides:

- Interactive chat interface with AI assistant
- Knowledge base management and document upload
- Problem solving with multi-agent reasoning
- Deep research reports with comprehensive analysis
- Notebook management for organizing content
- System monitoring and analytics
- Settings and configuration management
- History tracking for all activities

## 🏗️ Architecture

```
web/
├── app/                      # Next.js app directory
│   ├── page.tsx             # Home page with chat interface
│   ├── layout.tsx            # Root layout with sidebar
│   ├── globals.css           # Global styles and themes
│   ├── history/              # Activity history pages
│   ├── knowledge/            # Knowledge base management
│   ├── solver/               # Problem solving interface
│   ├── research/             # Research tools and reports
│   ├── notebook/             # Notebook management
│   ├── monitor/              # System monitoring dashboard
│   └── settings/             # Settings and configuration
├── components/               # React components
│   ├── Sidebar.tsx           # Main navigation sidebar
│   ├── RightSidebar.tsx      # Hub files and quick actions
│   ├── SystemStatus.tsx      # System health indicator
│   ├── ActivityDetail.tsx    # Activity detail modal
│   ├── ChatSessionDetail.tsx # Chat session viewer
│   ├── AddToNotebookModal.tsx # Add content to notebook
│   ├── NotebookImportModal.tsx # Import notebook records
│   ├── ThemeScript.tsx       # Theme initialization
│   ├── Mermaid.tsx           # Mermaid diagram renderer
│   ├── research/             # Research-specific components
│   │   ├── ActiveTaskDetail.tsx
│   │   ├── ResearchDashboard.tsx
│   │   └── TaskGrid.tsx
│   └── ui/                   # Reusable UI components
│       ├── Button.tsx
│       └── Modal.tsx
├── context/                  # React context providers
│   └── GlobalContext.tsx     # Global state management
├── hooks/                    # Custom React hooks
│   ├── useTheme.ts           # Theme management
│   ├── useQuestionReducer.ts # Question generation state
│   └── useResearchReducer.ts # Research workflow state
├── lib/                      # Utility libraries
│   ├── api.ts                # API client configuration
│   ├── i18n.ts               # Internationalization
│   ├── latex.ts              # LaTeX processing
│   ├── theme.ts              # Theme utilities
│   ├── debounce.ts           # Debounce utility
│   └── pdfExport.ts          # PDF export functionality
├── types/                    # TypeScript type definitions
│   └── research.ts           # Research-related types
└── public/                   # Static assets
    └── logo.png              # Application logo
```

## 🛠️ Technology Stack

- **Framework**: Next.js (App Router)
- **Runtime**: React 19
- **Language**: TypeScript
- **Styling**: Tailwind CSS with dark mode support
- **UI Components**: Custom components with Lucide React icons
- **Markdown**: react-markdown with KaTeX for mathematical expressions
- **PDF Export**: jsPDF + html2canvas for document generation
- **Charts**: Mermaid for diagram rendering
- **State Management**: React Context with custom hooks
- **Internationalization**: Built-in i18n support (English/Chinese)

## 📦 Key Features

### 🎨 Modern UI/UX
- **Dark/Light Theme**: Automatic theme switching with system preference detection
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Smooth Animations**: Micro-interactions and transitions for better UX
- **Accessibility**: WCAG compliant with keyboard navigation support

### 🔧 Core Functionality
- **Real-time Chat**: WebSocket-based chat with streaming responses
- **Knowledge Search**: Intelligent search with auto-suggestions
- **File Management**: Document upload, processing, and organization
- **Multi-language**: Support for English and Chinese interfaces
- **Export Options**: PDF, Markdown, and other format exports

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn package manager
- Backend API server running on port 8001

### Installation

```bash
cd web
npm install
```

### Development

```bash
npm run dev
```

The development server will start on `http://localhost:3000` with hot reload enabled.

### Build

```bash
npm run build
npm start
```

### Environment Configuration

Create a `.env.local` file in the web directory:

```bash
# API Configuration
NEXT_PUBLIC_API_BASE=http://localhost:8001
NEXT_PUBLIC_WS_BASE=ws://localhost:8001

# Optional: Custom port
PORT=3000
```

## 📁 Key Pages & Components

### 🏠 Home Page (app/page.tsx)
Interactive chat interface featuring:
- **Smart Chat**: Real-time conversation with AI assistant
- **Knowledge Integration**: RAG-enabled search with file suggestions
- **Web Search**: External information retrieval
- **Quick Actions**: Direct access to problem solving and research tools
- **Responsive Design**: Adaptive layout for different screen sizes

### 📚 Knowledge Base (app/knowledge/page.tsx)
Document management system with:
- **File Upload**: Drag-and-drop document upload with progress tracking
- **Knowledge Base Creation**: Organize documents into themed collections
- **Processing Status**: Real-time upload and processing feedback
- **Document Management**: View, delete, and organize uploaded files

### 🔬 Research Tools (app/research/page.tsx)
Comprehensive research platform including:
- **Multi-Agent Research**: Parallel research task execution
- **Research Dashboard**: Visual progress tracking with task grid
- **Report Generation**: Structured research reports with citations
- **Export Options**: PDF and Markdown export capabilities

### 📖 Notebook System (app/notebook/page.tsx)
Content organization and management:
- **Record Management**: Create, edit, and organize research records
- **Import/Export**: Cross-notebook record sharing and Markdown export
- **Search & Filter**: Advanced content discovery
- **Categorization**: Organize records by type and topic

### 📊 System Monitor (app/monitor/page.tsx)
Real-time system analytics:
- **Task Monitoring**: Track active and completed tasks
- **File Management**: Monitor file processing status
- **Token Usage**: Track API usage and costs
- **System Health**: Monitor system performance metrics

### ⚙️ Settings (app/settings/page.tsx)
Comprehensive configuration management:
- **LLM Providers**: Configure and test AI model providers
- **Environment Variables**: Manage system configuration
- **UI Preferences**: Theme, language, and interface settings
- **System Testing**: Health checks and service validation

## 🔌 API Integration

The frontend integrates with the backend API through both REST endpoints and WebSocket connections.

### REST API Examples

```typescript
import { apiUrl } from "@/lib/api";

// Fetch knowledge bases
const response = await fetch(`${apiUrl}/api/v1/knowledge/list`);
const knowledgeBases = await response.json();

// Upload documents
const formData = new FormData();
formData.append('files', file);
const uploadResponse = await fetch(`${apiUrl}/api/v1/knowledge/${kbName}/upload`, {
  method: 'POST',
  body: formData
});

// Get system status
const statusResponse = await fetch(`${apiUrl}/api/v1/settings/system/status`);
const systemStatus = await statusResponse.json();
```

### WebSocket Integration

```typescript
import { wsUrl } from "@/lib/api";

// Chat WebSocket
const chatWs = new WebSocket(`${wsUrl}/api/v1/chat/ws`);
chatWs.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'message') {
    // Handle streaming chat response
  }
};

// Research WebSocket
const researchWs = new WebSocket(`${wsUrl}/api/v1/research/ws`);
researchWs.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'progress') {
    // Update research progress
  }
};
```

## 🎨 Styling & Theming

### Tailwind CSS Configuration
The project uses Tailwind CSS with custom configuration for:
- **Color Palette**: Carefully selected colors for light and dark themes
- **Typography**: Optimized font scales and line heights
- **Spacing**: Consistent spacing system throughout the application
- **Animations**: Custom animations for smooth interactions

### Dark Mode Support
Comprehensive dark mode implementation:
- **System Preference Detection**: Automatically detects user's system theme
- **Manual Toggle**: Users can override system preference
- **Persistent Storage**: Theme preference saved in localStorage
- **Smooth Transitions**: Animated theme switching

### Global Styles (app/globals.css)
- **CSS Variables**: Dynamic color variables for theme switching
- **Base Styles**: Consistent typography and element styling
- **Utility Classes**: Custom utilities for common patterns
- **Component Styles**: Specialized styles for complex components

## 🗺️ Page Routes & Navigation

### Core Pages
- **`/`** - Home page with interactive chat interface
- **`/history`** - Activity history and chat session management
- **`/knowledge`** - Knowledge base and document management
- **`/notebook`** - Content organization and record management
- **`/settings`** - System configuration and preferences

### Specialized Tools
- **`/solver`** - Multi-agent problem solving interface
- **`/research`** - Deep research tools with comprehensive reporting
- **`/monitor`** - System monitoring and analytics dashboard

### Navigation Features
- **Collapsible Sidebar**: Space-efficient navigation with tooltips
- **Breadcrumb Navigation**: Clear page hierarchy indication
- **Quick Actions**: Contextual shortcuts to related functionality
- **Search Integration**: Global search across all content types

## ⚙️ Configuration & Customization

### API Configuration (lib/api.ts)
Centralized API endpoint management:

```typescript
export function apiUrl(path: string): string {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8001";
  return `${base}${path}`;
}

export function wsUrl(path: string): string {
  const base = process.env.NEXT_PUBLIC_WS_BASE || "ws://localhost:8001";
  return `${base}${path}`;
}
```

### Internationalization (lib/i18n.ts)
Multi-language support with:
- **Language Detection**: Automatic language preference detection
- **Translation Keys**: Structured translation key system
- **Dynamic Loading**: Efficient translation resource loading
- **Fallback Support**: Graceful fallback to default language

### Theme System (lib/theme.ts)
Advanced theming capabilities:
- **CSS Variable Management**: Dynamic theme variable updates
- **Theme Persistence**: localStorage-based theme preference storage
- **System Integration**: Automatic system theme detection
- **Smooth Transitions**: Animated theme switching

## 🔗 Integration & Architecture

### Backend Integration
- **API Server**: FastAPI backend (`src/api/`) providing REST and WebSocket endpoints
- **Database**: DuckDB integration (`src/db/`) for data persistence
- **Real-time Communication**: WebSocket connections for streaming responses

### State Management Architecture
- **Global Context**: Centralized state management with React Context
- **Custom Hooks**: Specialized hooks for complex state logic
- **Local Storage**: Persistent user preferences and settings
- **Real-time Updates**: WebSocket-driven state synchronization

## 🛠️ Development Guidelines

### Code Organization
```typescript
// Follow consistent import order
import React from 'react';
import { NextPage } from 'next';
import { useGlobal } from '@/context/GlobalContext';
import { apiUrl } from '@/lib/api';
import ComponentName from '@/components/ComponentName';
```

### Component Development
- **TypeScript First**: All components use TypeScript with proper typing
- **Responsive Design**: Mobile-first approach with Tailwind breakpoints
- **Accessibility**: ARIA labels, keyboard navigation, and screen reader support
- **Performance**: Lazy loading, memoization, and efficient re-renders

### Styling Standards
- **Tailwind Classes**: Use utility-first approach with consistent spacing
- **Dark Mode**: Always implement both light and dark variants
- **Animations**: Subtle transitions and micro-interactions
- **Consistency**: Follow established design patterns and color schemes

### Testing & Quality
- **Type Safety**: Comprehensive TypeScript coverage
- **Error Handling**: Graceful error states and user feedback
- **Loading States**: Proper loading indicators and skeleton screens
- **Performance**: Optimized bundle size and runtime performance

## 🚀 Deployment & Production

### Build Optimization
- **Static Generation**: Optimized static assets and pages
- **Code Splitting**: Automatic code splitting for better performance
- **Image Optimization**: Next.js Image component with automatic optimization
- **Bundle Analysis**: Built-in bundle analyzer for size optimization

### Environment Setup
```bash
# Production environment variables
NEXT_PUBLIC_API_BASE=https://your-api-domain.com
NEXT_PUBLIC_WS_BASE=wss://your-api-domain.com
NODE_ENV=production
```

### Performance Features
- **Lazy Loading**: Components and routes loaded on demand
- **Caching**: Intelligent caching strategies for API responses
- **Prefetching**: Automatic prefetching of likely navigation targets
- **Compression**: Gzip compression for all static assets

## ⚠️ Important Notes

### Security Considerations
- **Environment Variables**: Use `NEXT_PUBLIC_` prefix only for client-safe variables
- **API Security**: All API calls include proper error handling and validation
- **XSS Protection**: Sanitized user input and secure markdown rendering
- **CORS Configuration**: Backend CORS settings must allow frontend domain

### Browser Compatibility
- **Modern Browsers**: Optimized for Chrome, Firefox, Safari, and Edge
- **Progressive Enhancement**: Graceful degradation for older browsers
- **Mobile Support**: Full responsive design with touch-friendly interactions
- **Accessibility**: WCAG 2.1 AA compliance for screen readers and keyboard navigation

### Development Best Practices
- **Hot Reload**: Fast development with instant feedback
- **Error Boundaries**: Comprehensive error handling and user feedback
- **Type Safety**: Full TypeScript coverage with strict mode enabled
- **Code Quality**: ESLint and Prettier integration for consistent code style
