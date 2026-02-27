"use client";

import React, {
  createContext,
  useContext,
  useRef,
  useState,
  useEffect,
} from "react";
import { wsUrl, apiUrl } from "@/lib/api";
import {
  initializeTheme,
  setTheme as setThemeLib,
  getStoredTheme,
  type Theme,
} from "@/lib/theme";

// --- Types ---
interface LogEntry {
  type: string;
  content: string;
  timestamp?: number;
  level?: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  outputDir?: string;
}



// Question Progress Info
interface QuestionProgressInfo {
  stage:
    | "planning"
    | "researching"
    | "generating"
    | "validating"
    | "complete"
    // Mimic mode stages
    | "uploading"
    | "parsing"
    | "extracting"
    | null;
  progress: {
    current?: number;
    total?: number;
    round?: number;
    max_rounds?: number;
    status?: string;
  };
  // Parallel generation info
  subFocuses?: Array<{ id: string; focus: string; scenario_hint?: string }>;
  activeQuestions?: string[];
  completedQuestions?: number;
  failedQuestions?: number;
  extendedQuestions?: number; // Count of extended questions
}

// Question Agent Status
interface QuestionAgentStatus {
  [key: string]: "pending" | "running" | "done" | "error";
}

// Question Token Stats
interface QuestionTokenStats {
  model: string;
  calls: number;
  tokens: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

// Question State
interface QuestionState {
  isGenerating: boolean;
  logs: LogEntry[];
  questions: string[];
  topic: string;
  difficulty: string;
  type: string;
  count: number;
  selectedKb: string;
  agentStatus: QuestionAgentStatus;
  tokenStats: QuestionTokenStats;
  progress: QuestionProgressInfo;
  // Mimic mode specific
  mimicMode: boolean;
  uploadedFile: File | null;
  paperPath: string;
  maxQuestions: number;
}

// Query Info
interface QueryInfo {
  query: string;
  rationale?: string;
  iteration: number;
}




// Chat Types
interface ChatSource {
  rag?: Array<{ kb_name: string; content: string }>;
  web?: Array<{ url: string; title?: string; snippet?: string }>;
}

interface HomeChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource;
  isStreaming?: boolean;
  searchLogs?: Array<{
    level: string;
    message: string;
    timestamp: string;
    is_streaming?: boolean;  // Flag for streaming output (no timestamp prefix)
    task_id?: string;        // Task ID for grouping related streaming messages
    flush?: boolean;         // Flush flag for immediate output
  }>;
}

interface ChatState {
  sessionId: string | null;
  messages: HomeChatMessage[];
  isLoading: boolean;
  selectedKb: string;
  enableRag: boolean;
  enableWebSearch: boolean;
  searchMode: string;
  currentStage: string | null;
}

// Settings
interface Settings {
  theme: string;
  language: string;
}

interface GlobalContextType {
  // Question
  questionState: QuestionState;
  setQuestionState: React.Dispatch<React.SetStateAction<QuestionState>>;
  startQuestionGen: (
    topic: string,
    diff: string,
    type: string,
    count: number,
    kb: string,
  ) => void;
  startMimicQuestionGen: (
    file: File | null,
    paperPath: string,
    kb: string,
    maxQuestions?: number,
  ) => void;
  resetQuestionGen: () => void;

  // Chat
  chatState: ChatState;
  setChatState: React.Dispatch<React.SetStateAction<ChatState>>;
  sendChatMessage: (message: string) => void;
  stopChatMessage: () => void;
  loadChatSession: (sessionId: string) => void;
  clearChatHistory: () => void;
  newChatSession: () => void;

  // Theme
  theme: Theme;
  setTheme: (theme: Theme) => void;

  // Sidebar
  sidebarWidth: number;
  setSidebarWidth: (width: number) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;

  // Settings
  settings: Settings;
  refreshSettings: () => void;

  // UI Settings (alias for settings for backward compatibility)
  uiSettings: Settings;
}

const GlobalContext = createContext<GlobalContextType | undefined>(undefined);

export function GlobalProvider({ children }: { children: React.ReactNode }) {
  // --- Theme Logic ---
  const [theme, setThemeState] = useState<Theme>("light");
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedTheme = getStoredTheme();
      const finalTheme = storedTheme || initializeTheme();
      setThemeState(finalTheme);
      setIsInitialized(true);
    }
  }, []);

  const handleSetTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    setThemeLib(newTheme);
  };

  // --- Settings Logic ---
  const [settings, setSettings] = useState<Settings>({
    theme: "light",
    language: "en",
  });

  const refreshSettings = async () => {
    try {
      const response = await fetch(apiUrl("/api/v1/settings/ui"));
      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data) {
          const newSettings = {
            theme: result.data.theme || "light",
            language: result.data.language || "en",
          };
          setSettings(newSettings);
          
          // Apply theme immediately
          setThemeLib(newSettings.theme);
          
          // Store in localStorage for persistence
          localStorage.setItem("sirchmunk_settings", JSON.stringify(newSettings));
          
          console.log("Settings refreshed:", newSettings);
        }
      }
    } catch (error) {
      console.error("Failed to fetch settings:", error);
    }
  };

  useEffect(() => {
    if (isInitialized) {
      // Then fetch from backend and sync
      refreshSettings();
    }
  }, [isInitialized]);

  // --- Sidebar State ---
  const SIDEBAR_MIN_WIDTH = 64;
  const SIDEBAR_MAX_WIDTH = 320;
  const SIDEBAR_DEFAULT_WIDTH = 256;
  const SIDEBAR_COLLAPSED_WIDTH = 64;

  const [sidebarWidth, setSidebarWidthState] = useState<number>(
    SIDEBAR_DEFAULT_WIDTH,
  );
  const [sidebarCollapsed, setSidebarCollapsedState] = useState<boolean>(false);

  // Initialize sidebar state from localStorage
  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedWidth = localStorage.getItem("sidebarWidth");
      const storedCollapsed = localStorage.getItem("sidebarCollapsed");

      if (storedWidth) {
        const width = parseInt(storedWidth, 10);
        if (
          !isNaN(width) &&
          width >= SIDEBAR_MIN_WIDTH &&
          width <= SIDEBAR_MAX_WIDTH
        ) {
          setSidebarWidthState(width);
        }
      }

      if (storedCollapsed) {
        setSidebarCollapsedState(storedCollapsed === "true");
      }
    }
  }, []);

  const setSidebarWidth = (width: number) => {
    const clampedWidth = Math.max(
      SIDEBAR_MIN_WIDTH,
      Math.min(SIDEBAR_MAX_WIDTH, width),
    );
    setSidebarWidthState(clampedWidth);
    if (typeof window !== "undefined") {
      localStorage.setItem("sidebarWidth", clampedWidth.toString());
    }
  };

  const setSidebarCollapsed = (collapsed: boolean) => {
    setSidebarCollapsedState(collapsed);
    if (typeof window !== "undefined") {
      localStorage.setItem("sidebarCollapsed", collapsed.toString());
    }
  };

  // --- Question Logic ---
  const [questionState, setQuestionState] = useState<QuestionState>({
    isGenerating: false,
    logs: [],
    questions: [],
    topic: "",
    difficulty: "medium",
    type: "multiple_choice",
    count: 5,
    selectedKb: "",
    agentStatus: {},
    tokenStats: {
      model: "",
      calls: 0,
      tokens: 0,
      input_tokens: 0,
      output_tokens: 0,
      cost: 0,
    },
    progress: {
      stage: null,
      progress: {},
    },
    mimicMode: false,
    uploadedFile: null,
    paperPath: "",
    maxQuestions: 10,
  });

  const questionWs = useRef<WebSocket | null>(null);

  const startQuestionGen = (
    topic: string,
    diff: string,
    type: string,
    count: number,
    kb: string,
  ) => {
    if (questionWs.current) questionWs.current.close();

    setQuestionState((prev) => ({
      ...prev,
      isGenerating: true,
      logs: [],
      questions: [],
      topic,
      difficulty: diff,
      type,
      count,
      selectedKb: kb,
      mimicMode: false,
      agentStatus: {},
      tokenStats: {
        model: "",
        calls: 0,
        tokens: 0,
        input_tokens: 0,
        output_tokens: 0,
        cost: 0,
      },
      progress: {
        stage: null,
        progress: {},
      },
    }));

    const ws = new WebSocket(wsUrl("/api/v1/question/ws"));
    questionWs.current = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          topic,
          difficulty: diff,
          type,
          count,
          kb_name: kb,
        }),
      );
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "log") {
        setQuestionState((prev) => ({
          ...prev,
          logs: [...prev.logs, { type: "info", content: data.content }],
        }));
      } else if (data.type === "progress") {
        setQuestionState((prev) => ({
          ...prev,
          progress: {
            ...prev.progress,
            stage: data.stage,
            progress: data.progress || {},
            subFocuses: data.subFocuses,
            activeQuestions: data.activeQuestions,
            completedQuestions: data.completedQuestions,
            failedQuestions: data.failedQuestions,
            extendedQuestions: data.extendedQuestions,
          },
        }));
      } else if (data.type === "agent_status") {
        setQuestionState((prev) => ({
          ...prev,
          agentStatus: data.status,
        }));
      } else if (data.type === "token_stats") {
        setQuestionState((prev) => ({
          ...prev,
          tokenStats: data.stats,
        }));
      } else if (data.type === "questions") {
        setQuestionState((prev) => ({
          ...prev,
          questions: data.questions,
        }));
      } else if (data.type === "complete") {
        setQuestionState((prev) => ({
          ...prev,
          isGenerating: false,
          questions: data.questions || prev.questions,
          progress: {
            ...prev.progress,
            stage: "complete",
          },
        }));
        ws.close();
      } else if (data.type === "error") {
        setQuestionState((prev) => ({
          ...prev,
          isGenerating: false,
          logs: [
            ...prev.logs,
            { type: "error", content: `Error: ${data.content}` },
          ],
        }));
        ws.close();
      }
    };

    ws.onerror = () => {
      setQuestionState((prev) => ({
        ...prev,
        isGenerating: false,
        logs: [
          ...prev.logs,
          { type: "error", content: "WebSocket connection error" },
        ],
      }));
    };

    ws.onclose = () => {
      if (questionWs.current === ws) {
        questionWs.current = null;
      }
    };
  };

  const startMimicQuestionGen = (
    file: File | null,
    paperPath: string,
    kb: string,
    maxQuestions: number = 10,
  ) => {
    if (questionWs.current) questionWs.current.close();

    setQuestionState((prev) => ({
      ...prev,
      isGenerating: true,
      logs: [],
      questions: [],
      mimicMode: true,
      uploadedFile: file,
      paperPath,
      selectedKb: kb,
      maxQuestions,
      agentStatus: {},
      tokenStats: {
        model: "",
        calls: 0,
        tokens: 0,
        input_tokens: 0,
        output_tokens: 0,
        cost: 0,
      },
      progress: {
        stage: null,
        progress: {},
      },
    }));

    const ws = new WebSocket(wsUrl("/api/v1/question/mimic"));
    questionWs.current = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          paper_path: paperPath,
          kb_name: kb,
          max_questions: maxQuestions,
        }),
      );
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "log") {
        setQuestionState((prev) => ({
          ...prev,
          logs: [...prev.logs, { type: "info", content: data.content }],
        }));
      } else if (data.type === "progress") {
        setQuestionState((prev) => ({
          ...prev,
          progress: {
            ...prev.progress,
            stage: data.stage,
            progress: data.progress || {},
          },
        }));
      } else if (data.type === "questions") {
        setQuestionState((prev) => ({
          ...prev,
          questions: data.questions,
        }));
      } else if (data.type === "complete") {
        setQuestionState((prev) => ({
          ...prev,
          isGenerating: false,
          questions: data.questions || prev.questions,
          progress: {
            ...prev.progress,
            stage: "complete",
          },
        }));
        ws.close();
      } else if (data.type === "error") {
        setQuestionState((prev) => ({
          ...prev,
          isGenerating: false,
          logs: [
            ...prev.logs,
            { type: "error", content: `Error: ${data.content}` },
          ],
        }));
        ws.close();
      }
    };

    ws.onerror = () => {
      setQuestionState((prev) => ({
        ...prev,
        isGenerating: false,
        logs: [
          ...prev.logs,
          { type: "error", content: "WebSocket connection error" },
        ],
      }));
    };

    ws.onclose = () => {
      if (questionWs.current === ws) {
        questionWs.current = null;
      }
    };
  };

  const resetQuestionGen = () => {
    if (questionWs.current) {
      questionWs.current.close();
      questionWs.current = null;
    }
    setQuestionState((prev) => ({
      ...prev,
      isGenerating: false,
      logs: [],
      questions: [],
      topic: "",
      difficulty: "medium",
      type: "multiple_choice",
      count: 5,
      selectedKb: "",
      mimicMode: false,
      uploadedFile: null,
      paperPath: "",
      maxQuestions: 10,
      agentStatus: {},
      tokenStats: {
        model: "",
        calls: 0,
        tokens: 0,
        input_tokens: 0,
        output_tokens: 0,
        cost: 0,
      },
      progress: {
        stage: null,
        progress: {},
      },
    }));
  };


  // --- Chat Logic ---
  const [chatState, setChatState] = useState<ChatState>({
    sessionId: null,
    messages: [],
    isLoading: false,
    selectedKb: "",
    enableRag: false,
    enableWebSearch: false,
    searchMode: "FAST",
    currentStage: null,
  });
  const chatWs = useRef<WebSocket | null>(null);
  // Use ref to always have the latest sessionId in WebSocket callbacks (avoid closure issues)
  const sessionIdRef = useRef<string | null>(null);

  const sendChatMessage = (message: string) => {
    if (!message.trim() || chatState.isLoading) return;

    // Add user message and prepare assistant message for search logs
    setChatState((prev) => ({
      ...prev,
      isLoading: true,
      currentStage: "connecting",
      messages: [
        ...prev.messages, 
        { role: "user", content: message },
        { role: "assistant", content: "", isStreaming: true, searchLogs: [] }
      ],
    }));

    // Close existing connection if any
    if (chatWs.current) {
      chatWs.current.close();
    }

    const ws = new WebSocket(wsUrl("/api/v1/chat"));
    chatWs.current = ws;

    let assistantMessage = "";

    ws.onopen = () => {
      // Convert messages to simple format for history
      const history = chatState.messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      ws.send(
        JSON.stringify({
          message,
          // Use ref to get the latest sessionId (avoids closure capturing stale state)
          session_id: sessionIdRef.current,
          history,
          kb_name: chatState.selectedKb,
          enable_rag: chatState.enableRag,  // Use actual RAG state from chatState
          enable_web_search: chatState.enableWebSearch,
          search_mode: chatState.searchMode,
        }),
      );
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "session") {
        // Store session ID from backend - update both ref and state
        sessionIdRef.current = data.session_id;
        setChatState((prev) => ({
          ...prev,
          sessionId: data.session_id,
        }));
      } else if (data.type === "status") {
        setChatState((prev) => ({
          ...prev,
          currentStage: data.stage || data.message,
        }));
      } else if (data.type === "stream") {
        assistantMessage += data.content;
        setChatState((prev) => {
          const messages = [...prev.messages];
          const lastMessage = messages[messages.length - 1];
          if (lastMessage?.role === "assistant" && lastMessage?.isStreaming) {
            // Update existing streaming message
            messages[messages.length - 1] = {
              ...lastMessage,
              content: assistantMessage,
            };
          } else {
            // Add new streaming message
            messages.push({
              role: "assistant",
              content: assistantMessage,
              isStreaming: true,
            });
          }
          return { ...prev, messages, currentStage: "generating" };
        });
      } else if (data.type === "search_log") {
        // Handle search log streaming with smart merging and deduplication
        console.log("[DEBUG] Received search_log:", data);
        setChatState((prev) => {
          const messages = [...prev.messages];
          const lastMessage = messages[messages.length - 1];
          if (lastMessage?.role === "assistant") {
            const searchLogs = [...(lastMessage.searchLogs || [])];
            
            // Deduplication: Check if an identical non-streaming message already exists
            // (prevents duplicate logs from being added)
            if (!data.is_streaming && searchLogs.length > 0) {
              const isDuplicate = searchLogs.some(
                (log) => 
                  !log.is_streaming && 
                  log.level === data.level && 
                  log.message === data.message
              );
              if (isDuplicate) {
                console.log("[DEBUG] Skipping duplicate log:", data.message);
                return prev; // No state change
              }
            }
            
            // Smart merge: if this is a streaming message with same task_id as the last log,
            // append to the last log instead of creating a new one
            if (data.is_streaming && data.task_id && searchLogs.length > 0) {
              const lastLog = searchLogs[searchLogs.length - 1];
              if (lastLog.is_streaming && lastLog.task_id === data.task_id && lastLog.level === data.level) {
                // Merge with the last streaming log
                searchLogs[searchLogs.length - 1] = {
                  ...lastLog,
                  message: lastLog.message + data.message,
                  timestamp: data.timestamp, // Update to latest timestamp
                };
                console.log("[DEBUG] Merged streaming log:", searchLogs[searchLogs.length - 1]);
              } else {
                // Different task or level, add as new log
                searchLogs.push({
                  level: data.level,
                  message: data.message,
                  timestamp: data.timestamp,
                  is_streaming: data.is_streaming,
                  task_id: data.task_id,
                  flush: data.flush,
                });
              }
            } else {
              // Non-streaming or first streaming message, add as new log
              searchLogs.push({
                level: data.level,
                message: data.message,
                timestamp: data.timestamp,
                is_streaming: data.is_streaming,
                task_id: data.task_id,
                flush: data.flush,
              });
            }
            
            messages[messages.length - 1] = {
              ...lastMessage,
              searchLogs,
            };
            console.log("[DEBUG] Updated searchLogs count:", messages[messages.length - 1].searchLogs?.length);
          } else {
            console.log("[DEBUG] No assistant message found for search log");
          }
          return { ...prev, messages };
        });
      } else if (data.type === "search_complete") {
        // Handle search completion
        setChatState((prev) => ({
          ...prev,
          currentStage: "generating"
        }));
      } else if (data.type === "search_error") {
        // Handle search error
        setChatState((prev) => {
          const messages = [...prev.messages];
          const lastMessage = messages[messages.length - 1];
          if (lastMessage?.role === "assistant") {
            const searchLogs = lastMessage.searchLogs || [];
            messages[messages.length - 1] = {
              ...lastMessage,
              searchLogs: [...searchLogs, {
                level: "error",
                message: data.message,
                timestamp: new Date().toISOString()
              }],
            };
          }
          return { ...prev, messages };
        });
      } else if (data.type === "sources") {
        setChatState((prev) => {
          const messages = [...prev.messages];
          const lastMessage = messages[messages.length - 1];
          if (lastMessage?.role === "assistant") {
            messages[messages.length - 1] = {
              ...lastMessage,
              sources: { rag: data.rag, web: data.web },
            };
          }
          return { ...prev, messages };
        });
      } else if (data.type === "result") {
        setChatState((prev) => {
          const messages = [...prev.messages];
          const lastMessage = messages[messages.length - 1];
          if (lastMessage?.role === "assistant") {
            messages[messages.length - 1] = {
              ...lastMessage,
              content: data.content,
              isStreaming: false,
            };
          }
          return {
            ...prev,
            messages,
            isLoading: false,
            currentStage: null,
          };
        });
        ws.close();
      } else if (data.type === "error") {
        setChatState((prev) => ({
          ...prev,
          isLoading: false,
          currentStage: null,
          messages: [
            ...prev.messages,
            { role: "assistant", content: `Error: ${data.message}` },
          ],
        }));
        ws.close();
      }
    };

    ws.onerror = () => {
      setChatState((prev) => ({
        ...prev,
        isLoading: false,
        currentStage: null,
        messages: [
          ...prev.messages,
          { role: "assistant", content: "Connection error. Please try again." },
        ],
      }));
    };

    ws.onclose = () => {
      if (chatWs.current === ws) {
        chatWs.current = null;
      }
      // Ensure loading state is reset when connection closes
      setChatState((prev) => {
        // Only reset if still loading (don't override a successful completion)
        if (prev.isLoading) {
          return {
            ...prev,
            isLoading: false,
            currentStage: null,
          };
        }
        return prev;
      });
    };
  };

  const stopChatMessage = () => {
    if (chatWs.current) {
      chatWs.current.close();
      chatWs.current = null;
    }
    setChatState((prev) => {
      const messages = [...prev.messages];
      const lastMessage = messages[messages.length - 1];
      if (lastMessage?.role === "assistant" && lastMessage?.isStreaming) {
        messages[messages.length - 1] = {
          ...lastMessage,
          isStreaming: false,
          content: lastMessage.content || "(stopped)",
        };
      }
      return {
        ...prev,
        messages,
        isLoading: false,
        currentStage: null,
      };
    });
  };

  const loadChatSession = async (sessionId: string) => {
    try {
      const response = await fetch(
        apiUrl(`/api/v1/chat/sessions/${sessionId}`),
      );
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          const session = data.data;

          // Convert session messages to HomeChatMessage format
          const messages: HomeChatMessage[] = session.messages.map((msg: any) => ({
            role: msg.role,
            content: msg.content,
            sources: msg.sources,
            isStreaming: false,
          }));

          setChatState((prev) => ({
            ...prev,
            sessionId: session.session_id,
            messages,
            selectedKb: session.settings?.kb_name || prev.selectedKb,
            enableRag: session.settings?.enable_rag || prev.enableRag,
            enableWebSearch:
              session.settings?.enable_web_search || prev.enableWebSearch,
            searchMode: session.settings?.search_mode || prev.searchMode,
          }));

          // Update sessionId ref
          sessionIdRef.current = session.session_id;
        }
      }
    } catch (error) {
      console.error("Failed to load chat session:", error);
    }
  };

  const clearChatHistory = () => {
    setChatState((prev) => ({
      ...prev,
      messages: [],
    }));
  };

  const newChatSession = () => {
    setChatState((prev) => ({
      ...prev,
      sessionId: null,
      messages: [],
    }));
    sessionIdRef.current = null;
  };

  return (
    <GlobalContext.Provider
      value={{
        // Theme
        theme,
        setTheme: handleSetTheme,

        // Sidebar
        sidebarWidth,
        setSidebarWidth,
        sidebarCollapsed,
        setSidebarCollapsed,

        // Settings
        settings,
        refreshSettings,
        uiSettings: settings, // Alias for backward compatibility

        // Question
        questionState,
        setQuestionState,
        startQuestionGen,
        startMimicQuestionGen,
        resetQuestionGen,

        // Chat
        chatState,
        setChatState,
        sendChatMessage,
        stopChatMessage,
        loadChatSession,
        clearChatHistory,
        newChatSession,
      }}
    >
      {children}
    </GlobalContext.Provider>
  );
}

export function useGlobal() {
  const context = useContext(GlobalContext);
  if (!context) throw new Error("useGlobal must be used within GlobalProvider");
  return context;
}