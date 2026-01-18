"use client";

import { useState, useEffect, useRef } from "react";
import {
  Send,
  Loader2,
  Bot,
  User,
  Database,
  Globe,
  Calculator,
  FileText,
  Microscope,
  Lightbulb,
  Trash2,
  ExternalLink,
  BookOpen,
  Sparkles,
  Edit3,
  GraduationCap,
  PenTool,
  Search,
  ChevronDown,
} from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { useGlobal } from "@/context/GlobalContext";
import { apiUrl } from "@/lib/api";
import { processLatexContent } from "@/lib/latex";
import { getTranslation, type Language } from "@/lib/i18n";

interface KnowledgeBase {
  name: string;
  is_default?: boolean;
}

interface SearchSuggestion {
  filename: string;
  display_name: string;
  type: string;
  size: string;
  kb_name: string;
  highlight_start: number;
  highlight_end: number;
}

interface ChatState {
  sessionId: string | null;
  messages: any[];
  isLoading: boolean;
  selectedKb: string;
  enableRag: boolean;
  enableWebSearch: boolean;
  currentStage: string | null;
}

export default function HomePage() {
  const {
    chatState,
    setChatState,
    sendChatMessage,
    newChatSession,
    settings,
  } = useGlobal();
  const t = (key: string) => getTranslation((settings?.language as Language) || "en", key);


  const [inputMessage, setInputMessage] = useState("");
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [searchSuggestions, setSearchSuggestions] = useState<SearchSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const [showFileSelector, setShowFileSelector] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string>("");
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [showPathDropdown, setShowPathDropdown] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pathDropdownRef = useRef<HTMLDivElement>(null);

  // Fetch knowledge bases
  useEffect(() => {
    fetch(apiUrl("/api/v1/knowledge/list"))
      .then((res) => res.json())
      .then((response) => {
        // Handle API response structure
        const data = response.data || response;
        const kbList = Array.isArray(data) ? data : [];
        setKbs(kbList);
        if (!chatState.selectedKb && kbList.length > 0) {
          const defaultKb = kbList.find((kb: KnowledgeBase) => kb.is_default);
          if (defaultKb) {
            setChatState((prev: ChatState) => ({ ...prev, selectedKb: defaultKb.name }));
          } else {
            setChatState((prev: ChatState) => ({ ...prev, selectedKb: kbList[0].name }));
          }
        }
      })
      .catch((err) => console.error("Failed to fetch KBs:", err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatState.messages]);

  const handleSend = async () => {
    if (!inputMessage.trim() || chatState.isLoading) return;

    // Check for Google search shortcut
    const trimmedMessage = inputMessage.trim();
    if (trimmedMessage.toLowerCase().startsWith('g:') || trimmedMessage.toLowerCase().startsWith('G:')) {
      const searchQuery = trimmedMessage.slice(2).trim();
      if (searchQuery) {
        // Open Google search in new tab
        window.open(`https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`, '_blank');
      } else {
        // Open Google homepage if no query
        window.open('https://www.google.com', '_blank');
      }
      setInputMessage("");
      return;
    }

    // Check if File search is enabled and we have selected paths
    if (chatState.enableRag && selectedPaths.length > 0) {
      // Use WebSocket chat with search paths instead of direct REST API
      setChatState((prev) => ({
        ...prev,
        selectedKb: selectedPaths.join(","), // Pass paths as comma-separated string
      }));
    }

    sendChatMessage(inputMessage);
    setInputMessage("");
  };

  // Fetch search suggestions
  const fetchSearchSuggestions = async (query: string) => {
    if (!chatState.enableRag || !chatState.selectedKb || query.length < 2) {
      setSearchSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    try {
      const response = await fetch(
        apiUrl(`/api/v1/search/${chatState.selectedKb}/suggestions?query=${encodeURIComponent(query)}&limit=8`)
      );
      const result = await response.json();
      if (result.success) {
        setSearchSuggestions(result.data);
        setShowSuggestions(result.data.length > 0);
        setSelectedSuggestionIndex(-1);
      }
    } catch (error) {
      console.error("Failed to fetch suggestions:", error);
      setSearchSuggestions([]);
      setShowSuggestions(false);
    }
  };

  // Debounced search suggestions
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchSearchSuggestions(inputMessage);
    }, 200);
    return () => clearTimeout(timer);
  }, [inputMessage, chatState.enableRag, chatState.selectedKb]);

  // Click outside to close suggestions and dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
        setSelectedSuggestionIndex(-1);
      }

      if (
        pathDropdownRef.current &&
        !pathDropdownRef.current.contains(event.target as Node) &&
        !(event.target as Element).closest('button')?.textContent?.includes('Select Path')
      ) {
        setShowPathDropdown(false);
      }
    };

    if (showSuggestions || showPathDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showSuggestions, showPathDropdown]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showSuggestions && searchSuggestions.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedSuggestionIndex(prev =>
          prev < searchSuggestions.length - 1 ? prev + 1 : 0
        );
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedSuggestionIndex(prev =>
          prev > 0 ? prev - 1 : searchSuggestions.length - 1
        );
      } else if (e.key === "Tab" && selectedSuggestionIndex >= 0) {
        e.preventDefault();
        const suggestion = searchSuggestions[selectedSuggestionIndex];
        setInputMessage(`Search in ${suggestion.filename}: `);
        setShowSuggestions(false);
        setSelectedSuggestionIndex(-1);
      } else if (e.key === "Escape") {
        setShowSuggestions(false);
        setSelectedSuggestionIndex(-1);
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (showSuggestions && selectedSuggestionIndex >= 0) {
        const suggestion = searchSuggestions[selectedSuggestionIndex];
        setInputMessage(`Search in ${suggestion.filename}: `);
        setShowSuggestions(false);
        setSelectedSuggestionIndex(-1);
        inputRef.current?.focus();
      } else {
        handleSend();
      }
    }
  };

  const handleSuggestionClick = (suggestion: SearchSuggestion) => {
    setInputMessage(`Search in ${suggestion.filename}: `);
    setShowSuggestions(false);
    setSelectedSuggestionIndex(-1);
    inputRef.current?.focus();
  };

  const highlightMatch = (text: string, start: number, end: number) => {
    if (start < 0 || end <= start) return text;
    return (
      <>
        {text.slice(0, start)}
        <span className="bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100 px-0.5 rounded">
          {text.slice(start, end)}
        </span>
        {text.slice(end)}
      </>
    );
  };

  const quickActions = [
    {
      icon: Calculator,
      label: t("Smart Problem Solving"),
      href: "/solver",
      color: "blue",
      description: "Multi-agent reasoning",
    },
    {
      icon: Microscope,
      label: t("Deep Research Reports"),
      href: "/research",
      color: "emerald",
      description: "Comprehensive analysis",
    },
  ];

  const hasMessages = chatState.messages.length > 0;

  return (
    <div className="h-screen flex animate-fade-in">
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
      {/* Empty State / Welcome Screen */}
      {!hasMessages && (
        <div className="flex-1 flex flex-col items-center justify-center px-6">
          <div className="text-center max-w-2xl mx-auto mb-8">
            <div className="flex items-center justify-center gap-4 mb-3">
              <Image
                src="/logo-v1.png"
                alt="Sirchmunk Logo"
                width={56}
                height={56}
                className="object-contain"
                priority
              />
              <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 tracking-tight">
                {t("Welcome to Sirchmunk")}
              </h1>
            </div>
            <p className="text-lg text-slate-500 dark:text-slate-400">
              {t("How can I help you today?")}
            </p>
          </div>

          {/* Input Box - Centered */}
          <div className="w-full max-w-2xl mx-auto mb-12">
            {/* Mode Toggles */}
            <div className="flex items-center justify-between mb-3 px-1">
              <div className="flex items-center gap-2">
                {/* File Toggle */}
                <button
                  onClick={() => {
                    if (!chatState.enableRag) {
                      setShowFileSelector(true);
                    } else {
                      setChatState((prev) => ({
                        ...prev,
                        enableRag: false,
                      }));
                      setSelectedPath("");
                    }
                  }}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    chatState.enableRag
                      ? "bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-700"
                      : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700"
                  }`}
                >
                  <Database className="w-3.5 h-3.5" />
                  {t("FileSystem")}
                </button>

                {/* Web Search Toggle */}
                <button
                  onClick={() =>
                    setChatState((prev) => ({
                      ...prev,
                      enableWebSearch: !prev.enableWebSearch,
                    }))
                  }
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    chatState.enableWebSearch
                      ? "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700"
                      : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700"
                  }`}
                >
                  <Globe className="w-3.5 h-3.5" />
                  {t("WebSearch")}
                </button>
              </div>

              {/* Selected Path Display */}
              {chatState.enableRag && selectedPath && (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-lg text-sm">
                  <span className="text-blue-700 dark:text-blue-300 font-medium truncate max-w-[200px]">
                    {selectedPath.split('/').pop() || selectedPath}
                  </span>
                  <button
                    onClick={() => {
                      setSelectedPath("");
                      setChatState((prev) => ({ ...prev, enableRag: false }));
                    }}
                    className="text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-200"
                  >
                    ×
                  </button>
                </div>
              )}
            </div>

            {/* File Selector Modal */}
            {showFileSelector && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999]">
                <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">
                    Select File or Folder
                  </h3>

                  <div className="space-y-4">
                    {/* File Input */}
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Choose Files (Multiple Selection Supported)
                      </label>
                      <div className="space-y-2">
                        <input
                          ref={fileInputRef}
                          type="file"
                          multiple
                          className="w-full text-sm text-slate-500 dark:text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 dark:file:bg-blue-900/30 dark:file:text-blue-300 dark:hover:file:bg-blue-900/50"
                          onChange={async (e) => {
                            const files = e.target.files;
                            if (files && files.length > 0) {
                              if (files.length === 1) {
                                const file = files[0];
                                // Try to get the full path using File System Access API
                                try {
                                  // @ts-ignore - File System Access API may not be in types yet
                                  if (file.webkitRelativePath) {
                                    setSelectedPath(file.webkitRelativePath);
                                  } else if ((file as any).path) {
                                    // @ts-ignore - Electron or Node.js environment
                                    setSelectedPath((file as any).path);
                                  } else {
                                    // Fallback: show file name with note about path limitation
                                    setSelectedPath(`${file.name} (browser security limits full path access)`);
                                  }
                                } catch (error) {
                                  setSelectedPath(`${file.name} (path access limited)`);
                                }
                              } else {
                                setSelectedPath(`${files.length} files selected`);
                              }
                            }
                          }}
                        />

                        {/* Native File Picker using Backend API */}
                        <button
                          type="button"
                          onClick={async () => {
                            try {
                              const response = await fetch(apiUrl("/api/v1/file-picker"), {
                                method: "POST",
                                headers: {
                                  "Content-Type": "application/json",
                                },
                                body: JSON.stringify({
                                  type: "files",
                                  multiple: true
                                }),
                              });

                              const result = await response.json();

                              if (result.success && result.data.paths.length > 0) {
                                if (result.data.paths.length === 1) {
                                  setSelectedPath(result.data.paths[0]);
                                } else {
                                  setSelectedPath(`${result.data.paths.length} files selected (native picker)`);
                                  // Store all paths for later use
                                  const newPaths = result.data.paths.filter((path: string) => !selectedPaths.includes(path));
                                  if (newPaths.length > 0) {
                                    setSelectedPaths(prev => [...prev, ...newPaths]);
                                  }
                                }
                              } else {
                                alert(result.error || 'Failed to open native file picker. Please use manual path input below.');
                              }
                            } catch (error) {
                              console.error('Error calling native file picker:', error);
                              alert('Failed to open native file picker. Please use manual path input below.');
                            }
                          }}
                          className="w-full px-3 py-2 text-sm bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700 rounded-lg hover:bg-emerald-100 dark:hover:bg-emerald-900/50 transition-colors"
                        >
                          🖥️ Native File Picker (Real Absolute Paths)
                        </button>

                        {/* Native Directory Picker using Backend API */}
                        <button
                          type="button"
                          onClick={async () => {
                            try {
                              const response = await fetch(apiUrl("/api/v1/file-picker"), {
                                method: "POST",
                                headers: {
                                  "Content-Type": "application/json",
                                },
                                body: JSON.stringify({
                                  type: "directory",
                                  multiple: false
                                }),
                              });

                              const result = await response.json();

                              if (result.success && result.data.paths.length > 0) {
                                const dirPath = result.data.paths[0];
                                setSelectedPath(dirPath);
                              } else {
                                alert(result.error || 'Failed to open native directory picker. Please use manual path input below.');
                              }
                            } catch (error) {
                              console.error('Error calling native directory picker:', error);
                              alert('Failed to open native directory picker. Please use manual path input below.');
                            }
                          }}
                          className="w-full px-3 py-2 text-sm bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-700 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
                        >
                          📂 Native Directory Picker (Real Absolute Paths)
                        </button>
                      </div>
                    </div>

                    {/* Manual Path Input */}
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Or Enter Path
                      </label>
                      <input
                        type="text"
                        placeholder="C:\Users\YourName\Documents\file.txt or /Users/yourname/Documents/file.txt"
                        value={selectedPath}
                        onChange={(e) => {
                          // Accept any path format - Windows, macOS, or Linux
                          setSelectedPath(e.target.value);
                        }}
                        className="w-full px-3 py-2 border border-slate-200 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
                      />
                    </div>
                  </div>

                  <div className="flex gap-3 mt-6">
                    <button
                      onClick={() => {
                        setShowFileSelector(false);
                        setSelectedPath("");
                        if (fileInputRef.current) {
                          fileInputRef.current.value = "";
                        }
                      }}
                      className="flex-1 px-4 py-2 text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 border border-slate-200 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => {
                        if (selectedPath.trim()) {
                          const files = fileInputRef.current?.files;
                          if (files && files.length > 0) {
                            // Handle multiple files - use real absolute paths
                            const newPaths = Array.from(files).map(file => {
                              // Try to get real path, fallback to file name
                              return file.webkitRelativePath || file.name;
                            });
                            const uniquePaths = newPaths.filter(path => !selectedPaths.includes(path));
                            if (uniquePaths.length > 0) {
                              setSelectedPaths(prev => [...prev, ...uniquePaths]);
                              setChatState((prev) => ({
                                ...prev,
                                enableRag: true,
                                selectedKb: uniquePaths[0],
                              }));
                            }
                          } else if (selectedPath.trim() && !selectedPath.includes('files selected')) {
                            // Handle manual path input - accept any absolute path format
                            const newPath = selectedPath.trim();
                            if (!selectedPaths.includes(newPath)) {
                              setSelectedPaths(prev => [...prev, newPath]);
                              setChatState((prev) => ({
                                ...prev,
                                enableRag: true,
                                selectedKb: newPath,
                              }));
                            }
                          }
                          setShowFileSelector(false);
                        }
                      }}
                      disabled={!selectedPath.trim()}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      Select
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Input Field */}
            <div className="relative">
              <input
                ref={inputRef}
                type="text"
                className="w-full px-5 py-4 pr-14 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-500 text-slate-700 dark:text-slate-200 shadow-lg shadow-slate-200/50 dark:shadow-slate-900/50"
                placeholder={t("Ask anything...")}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={chatState.isLoading}
              />
              <button
                onClick={handleSend}
                disabled={chatState.isLoading || !inputMessage.trim()}
                className="absolute right-2 top-2 bottom-2 aspect-square bg-blue-600 text-white rounded-xl flex items-center justify-center hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all shadow-md shadow-blue-500/20"
              >
                {chatState.isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>

              {/* Search Suggestions Dropdown */}
              {showSuggestions && searchSuggestions.length > 0 && (
                <div
                  ref={suggestionsRef}
                  className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl shadow-xl shadow-slate-200/50 dark:shadow-slate-900/50 z-50 max-h-80 overflow-y-auto"
                >
                  <div className="p-2">
                    <div className="text-xs text-slate-500 dark:text-slate-400 px-3 py-2 border-b border-slate-100 dark:border-slate-700">
                      Found {searchSuggestions.length} file{searchSuggestions.length !== 1 ? 's' : ''} in {chatState.selectedKb}
                    </div>
                    {searchSuggestions.map((suggestion, index) => (
                      <button
                        key={index}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className={`w-full text-left px-3 py-3 rounded-lg transition-all duration-150 flex items-center gap-3 group ${
                          index === selectedSuggestionIndex
                            ? "bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700"
                            : "hover:bg-slate-50 dark:hover:bg-slate-700/50"
                        }`}
                      >
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-medium ${
                          suggestion.type.toLowerCase() === 'pdf'
                            ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                            : suggestion.type.toLowerCase() === 'docx'
                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                            : suggestion.type.toLowerCase() === 'pptx'
                            ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400'
                            : suggestion.type.toLowerCase() === 'csv'
                            ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                            : suggestion.type.toLowerCase() === 'xlsx'
                            ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400'
                            : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                        }`}>
                          {suggestion.type}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-slate-900 dark:text-slate-100 text-sm truncate">
                            {highlightMatch(suggestion.display_name, suggestion.highlight_start, suggestion.highlight_end)}
                          </div>
                          <div className="text-xs text-slate-500 dark:text-slate-400 truncate mt-0.5">
                            {suggestion.filename} • {suggestion.size}
                          </div>
                        </div>
                        <div className="text-xs text-slate-400 dark:text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity">
                          Tab to select
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      )}

      {/* Chat Interface - When there are messages */}
      {hasMessages && (
        <>
          {/* Header Bar */}
          <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              {/* Mode Toggles */}
              <button
                onClick={() => {
                  if (!chatState.enableRag) {
                    // Always show file selector when enabling RAG
                    setShowFileSelector(true);
                  } else {
                    // Disable RAG
                    setChatState((prev) => ({
                      ...prev,
                      enableRag: false,
                    }));
                  }
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  chatState.enableRag
                    ? "bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
                }`}
              >
                <Database className="w-3 h-3" />
                {t("FileSystem")}
              </button>

              <button
                onClick={() =>
                  setChatState((prev) => ({
                    ...prev,
                    enableWebSearch: !prev.enableWebSearch,
                  }))
                }
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  chatState.enableWebSearch
                    ? "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
                }`}
              >
                <Globe className="w-3 h-3" />
                {t("WebSearch")}
              </button>

              {chatState.enableRag && (
                <div className="relative">
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setShowPathDropdown(!showPathDropdown)}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors dark:text-slate-200"
                    >
                      <span className="max-w-[120px] truncate">
                        {chatState.selectedKb ? (chatState.selectedKb.split('/').pop() || chatState.selectedKb) : 'Select Path'}
                      </span>
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => setShowFileSelector(true)}
                      className="flex items-center justify-center w-6 h-6 text-xs bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400 rounded hover:bg-blue-200 dark:hover:bg-blue-900/70 transition-colors"
                      title="Add new path"
                    >
                      +
                    </button>
                  </div>

                  {showPathDropdown && selectedPaths.length > 0 && (
                    <div
                      ref={pathDropdownRef}
                      className="absolute top-full left-0 mt-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-50 min-w-[200px]"
                    >
                      {selectedPaths.map((path, index) => (
                        <div key={index} className="flex items-center justify-between px-3 py-2 hover:bg-slate-50 dark:hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg">
                          <button
                            onClick={() => {
                              setChatState((prev) => ({
                                ...prev,
                                selectedKb: path,
                              }));
                              setShowPathDropdown(false);
                            }}
                            className="flex-1 text-left text-xs text-slate-700 dark:text-slate-300 truncate"
                            title={path}
                          >
                            {path.split('/').pop() || path}
                          </button>
                          <button
                            onClick={() => {
                              const newPaths = selectedPaths.filter((_, i) => i !== index);
                              setSelectedPaths(newPaths);
                              if (chatState.selectedKb === path) {
                                setChatState((prev) => ({
                                  ...prev,
                                  selectedKb: newPaths[0] || "",
                                  enableRag: newPaths.length > 0,
                                }));
                              }
                            }}
                            className="ml-2 text-slate-400 hover:text-red-500 dark:text-slate-500 dark:hover:text-red-400"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <button
              onClick={newChatSession}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-500 dark:text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-lg transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              {t("New Chat")}
            </button>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
            {chatState.messages.map((msg, idx) => (
              <div
                key={idx}
                className="flex gap-4 w-full max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-2"
              >
                {msg.role === "user" ? (
                  <>
                    <div className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center shrink-0">
                      <User className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                    </div>
                    <div className="flex-1 bg-slate-100 dark:bg-slate-700 px-4 py-3 rounded-2xl rounded-tl-none text-slate-800 dark:text-slate-200">
                      {msg.content}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg shadow-blue-500/30">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="flex-1 space-y-3">
                      <div className="bg-white dark:bg-slate-800 px-5 py-4 rounded-2xl rounded-tl-none border border-slate-200 dark:border-slate-700 shadow-sm">
                        <div className="prose prose-slate dark:prose-invert prose-sm max-w-none">
                          <ReactMarkdown
                            remarkPlugins={[remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                          >
                            {processLatexContent(msg.content)}
                          </ReactMarkdown>
                        </div>

                        {/* Loading indicator */}
                        {msg.isStreaming && (
                          <div className="flex items-center gap-2 mt-3 text-blue-600 dark:text-blue-400 text-sm">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>{t("Generating response...")}</span>
                          </div>
                        )}

                        {/* Search Logs */}
                        {msg.searchLogs && msg.searchLogs.length > 0 && (
                          <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
                            <div className="flex items-center gap-2 mb-2">
                              <Search className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Search Process</span>
                            </div>
                            <div 
                              className="space-y-1 max-h-48 overflow-y-auto"
                              ref={(el) => {
                                // Auto-scroll to bottom when new logs arrive
                                if (el && msg.isStreaming) {
                                  el.scrollTop = el.scrollHeight;
                                }
                              }}
                            >
                              {msg.searchLogs.map((log, logIndex) => (
                                <div
                                  key={logIndex}
                                  className={`text-xs px-2 py-1 rounded font-mono ${
                                    log.level === 'error'
                                      ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                                      : log.level === 'warning'
                                      ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300'
                                      : log.level === 'success'
                                      ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                                      : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                                  }`}
                                >
                                  {/* Only show timestamp for non-streaming messages */}
                                  {!log.is_streaming && (
                                    <span className="opacity-60 mr-2">
                                      {new Date(log.timestamp).toLocaleTimeString()}
                                    </span>
                                  )}
                                  <span style={{ whiteSpace: log.is_streaming ? 'pre' : 'pre-wrap' }}>
                                    {log.message}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Sources */}
                      {msg.sources &&
                        (msg.sources.rag?.length ?? 0) +
                          (msg.sources.web?.length ?? 0) >
                          0 && (
                          <div className="flex flex-wrap gap-2">
                            {msg.sources.rag?.map((source, i) => (
                              <div
                                key={`rag-${i}`}
                                className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-lg text-xs"
                              >
                                <BookOpen className="w-3 h-3" />
                                <span>{source.kb_name}</span>
                              </div>
                            ))}
                            {msg.sources.web?.slice(0, 3).map((source, i) => (
                              <a
                                key={`web-${i}`}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg text-xs hover:bg-emerald-100 dark:hover:bg-emerald-900/50 transition-colors"
                              >
                                <Globe className="w-3 h-3" />
                                <span className="max-w-[150px] truncate">
                                  {source.title || source.url}
                                </span>
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            ))}
                          </div>
                        )}
                    </div>
                  </>
                )}
              </div>
            ))}

            {/* Status indicator */}
            {chatState.isLoading && chatState.currentStage && (
              <div className="flex gap-4 w-full max-w-4xl mx-auto">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shrink-0">
                  <Loader2 className="w-4 h-4 text-white animate-spin" />
                </div>
                <div className="flex-1 bg-slate-100 dark:bg-slate-800 px-4 py-3 rounded-2xl rounded-tl-none">
                  <div className="flex items-center gap-2 text-slate-600 dark:text-slate-300 text-sm">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                    </span>
                    {chatState.currentStage === "rag" &&
                      t("Searching knowledge base...")}
                    {chatState.currentStage === "web" &&
                      t("Searching the web...")}
                    {chatState.currentStage === "generating" &&
                      t("Generating response...")}
                    {!["rag", "web", "generating"].includes(
                      chatState.currentStage,
                    ) && chatState.currentStage}
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Input Area - Fixed at bottom */}
          <div className="border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-6 py-4">
            <div className="max-w-4xl mx-auto relative">
              <input
                ref={inputRef}
                type="text"
                className="w-full px-5 py-3.5 pr-14 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-500 text-slate-700 dark:text-slate-200"
                placeholder={t("Type your message...")}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={chatState.isLoading}
              />
              <button
                onClick={handleSend}
                disabled={chatState.isLoading || !inputMessage.trim()}
                className="absolute right-2 top-2 bottom-2 aspect-square bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all"
              >
                {chatState.isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        </>
      )}
      </div>

    </div>
  );
}
