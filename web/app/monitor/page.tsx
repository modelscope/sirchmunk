"use client";

import { useState, useEffect } from "react";
import {
  Monitor,
  Activity,
  FileText,
  Zap,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  Play,
  Pause,
  RotateCcw,
  TrendingUp,
  Database,
  Server,
  Cpu,
  HardDrive,
  BarChart3,
  PieChart,
  LineChart,
  Calendar,
  Filter,
  Download,
  RefreshCw,
} from "lucide-react";
import { apiUrl } from "@/lib/api";
import { getTranslation, type Language } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";

interface TaskStatus {
  id: string;
  type: "solve" | "research" | "chat" | "upload";
  name: string;
  status: "running" | "completed" | "failed" | "pending";
  progress: number;
  startTime: string;
  endTime?: string;
  duration?: number;
  tokensUsed: number;
  filesProcessed: number;
  error?: string;
}

interface FileInfo {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadTime: string;
  processedTime?: string;
  status: "uploaded" | "processing" | "processed" | "failed";
  tokensGenerated: number;
  associatedTasks: string[];
}

interface TokenUsage {
  model: string;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  cost: number;
  requests: number;
  lastUsed: string;
}

interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  activeConnections: number;
  uptime: string;
  lastUpdated: string;
}

export default function MonitorPage() {
  const { uiSettings } = useGlobal();
  const t = (key: string) => getTranslation((uiSettings?.language || "en") as Language, key);

  const [tasks, setTasks] = useState<TaskStatus[]>([]);
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"tasks" | "files" | "tokens" | "system">("tasks");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [filterStatus, setFilterStatus] = useState<string>("all");

  // Mock data generation
  const generateMockData = () => {
    const mockTasks: TaskStatus[] = [
      {
        id: "task_001",
        type: "solve",
        name: "Machine Learning Problem Analysis",
        status: "running",
        progress: 65,
        startTime: new Date(Date.now() - 300000).toISOString(),
        tokensUsed: 1250,
        filesProcessed: 3,
      },
      {
        id: "task_002",
        type: "research",
        name: "Climate Change Research Report",
        status: "completed",
        progress: 100,
        startTime: new Date(Date.now() - 1800000).toISOString(),
        endTime: new Date(Date.now() - 300000).toISOString(),
        duration: 1500,
        tokensUsed: 3420,
        filesProcessed: 8,
      },
      {
        id: "task_003",
        type: "chat",
        name: "AI Assistant Conversation",
        status: "completed",
        progress: 100,
        startTime: new Date(Date.now() - 600000).toISOString(),
        endTime: new Date(Date.now() - 120000).toISOString(),
        duration: 480,
        tokensUsed: 890,
        filesProcessed: 0,
      },
      {
        id: "task_004",
        type: "upload",
        name: "Document Processing",
        status: "failed",
        progress: 45,
        startTime: new Date(Date.now() - 900000).toISOString(),
        tokensUsed: 156,
        filesProcessed: 1,
        error: "File format not supported",
      },
    ];

    const mockFiles: FileInfo[] = [
      {
        id: "file_001",
        name: "research_paper.pdf",
        type: "PDF",
        size: 2048576,
        uploadTime: new Date(Date.now() - 3600000).toISOString(),
        processedTime: new Date(Date.now() - 3300000).toISOString(),
        status: "processed",
        tokensGenerated: 1250,
        associatedTasks: ["task_002"],
      },
      {
        id: "file_002",
        name: "dataset.csv",
        type: "CSV",
        size: 1024000,
        uploadTime: new Date(Date.now() - 1800000).toISOString(),
        processedTime: new Date(Date.now() - 1500000).toISOString(),
        status: "processed",
        tokensGenerated: 890,
        associatedTasks: ["task_001"],
      },
      {
        id: "file_003",
        name: "presentation.pptx",
        type: "PPTX",
        size: 5120000,
        uploadTime: new Date(Date.now() - 600000).toISOString(),
        status: "processing",
        tokensGenerated: 0,
        associatedTasks: [],
      },
    ];

    const mockTokenUsage: TokenUsage[] = [
      {
        model: "gpt-4",
        totalTokens: 15420,
        inputTokens: 8230,
        outputTokens: 7190,
        cost: 0.462,
        requests: 23,
        lastUsed: new Date(Date.now() - 120000).toISOString(),
      },
      {
        model: "gpt-3.5-turbo",
        totalTokens: 8950,
        inputTokens: 4200,
        outputTokens: 4750,
        cost: 0.0179,
        requests: 45,
        lastUsed: new Date(Date.now() - 300000).toISOString(),
      },
      {
        model: "text-embedding-ada-002",
        totalTokens: 25600,
        inputTokens: 25600,
        outputTokens: 0,
        cost: 0.0026,
        requests: 128,
        lastUsed: new Date(Date.now() - 180000).toISOString(),
      },
    ];

    const mockSystemMetrics: SystemMetrics = {
      cpuUsage: 45.2,
      memoryUsage: 68.7,
      diskUsage: 34.1,
      activeConnections: 12,
      uptime: "2d 14h 32m",
      lastUpdated: new Date().toISOString(),
    };

    setTasks(mockTasks);
    setFiles(mockFiles);
    setTokenUsage(mockTokenUsage);
    setSystemMetrics(mockSystemMetrics);
    setLoading(false);
  };

  useEffect(() => {
    generateMockData();
    
    if (autoRefresh) {
      const interval = setInterval(generateMockData, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
      case "processing":
        return <Play className="w-4 h-4 text-blue-500 animate-pulse" />;
      case "completed":
      case "processed":
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-500" />;
      case "pending":
      case "uploaded":
        return <Clock className="w-4 h-4 text-yellow-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
      case "processing":
        return "text-blue-600 bg-blue-100 dark:text-blue-400 dark:bg-blue-900/30";
      case "completed":
      case "processed":
        return "text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30";
      case "failed":
        return "text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30";
      case "pending":
      case "uploaded":
        return "text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/30";
      default:
        return "text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/30";
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const filteredTasks = tasks.filter(task => 
    filterStatus === "all" || task.status === filterStatus
  );

  const totalTokensUsed = tokenUsage.reduce((sum, usage) => sum + usage.totalTokens, 0);
  const totalCost = tokenUsage.reduce((sum, usage) => sum + usage.cost, 0);
  const activeTasks = tasks.filter(task => task.status === "running").length;
  const completedTasks = tasks.filter(task => task.status === "completed").length;

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-slate-500 dark:text-slate-400">{t("Loading monitor data...")}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col animate-fade-in p-6">
      {/* Header */}
      <div className="shrink-0 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 tracking-tight flex items-center gap-3">
              <Monitor className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              {t("System Monitor")}
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-2">
              {t("Real-time monitoring of tasks, files, and system resources")}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                autoRefresh
                  ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"
              }`}
            >
              {autoRefresh ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {autoRefresh ? t("Auto Refresh On") : t("Auto Refresh Off")}
            </button>
            <button
              onClick={generateMockData}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              {t("Refresh")}
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{t("Active Tasks")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{activeTasks}</p>
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{t("Completed")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{completedTasks}</p>
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{t("Total Tokens")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{totalTokensUsed.toLocaleString()}</p>
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{t("Total Cost")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">${totalCost.toFixed(3)}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 mt-6 bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
          {[
            { id: "tasks", label: t("Tasks"), icon: Activity },
            { id: "files", label: t("Files"), icon: FileText },
            { id: "tokens", label: t("Tokens"), icon: Zap },
            { id: "system", label: t("System"), icon: Server },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? "bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 shadow-sm"
                  : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {activeTab === "tasks" && (
          <div className="space-y-4">
            {/* Filter */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-slate-400" />
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 outline-none focus:border-blue-400 dark:text-slate-200"
                >
                  <option value="all">{t("All Status")}</option>
                  <option value="running">{t("Running")}</option>
                  <option value="completed">{t("Completed")}</option>
                  <option value="failed">{t("Failed")}</option>
                  <option value="pending">{t("Pending")}</option>
                </select>
              </div>
            </div>

            {/* Tasks List */}
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="divide-y divide-slate-100 dark:divide-slate-700">
                {filteredTasks.map((task) => (
                  <div key={task.id} className="p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        {getStatusIcon(task.status)}
                        <div>
                          <h3 className="font-semibold text-slate-900 dark:text-slate-100">{task.name}</h3>
                          <div className="flex items-center gap-4 mt-1 text-sm text-slate-500 dark:text-slate-400">
                            <span className="capitalize">{task.type}</span>
                            <span>{new Date(task.startTime).toLocaleString()}</span>
                            {task.duration && <span>{formatDuration(task.duration)}</span>}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(task.status)}`}>
                          {task.status}
                        </span>
                        <div className="flex items-center gap-4 mt-2 text-sm text-slate-500 dark:text-slate-400">
                          <span>{task.tokensUsed} tokens</span>
                          <span>{task.filesProcessed} files</span>
                        </div>
                      </div>
                    </div>
                    {task.status === "running" && (
                      <div className="mt-3">
                        <div className="flex justify-between text-sm text-slate-500 dark:text-slate-400 mb-1">
                          <span>{t("Progress")}</span>
                          <span>{task.progress}%</span>
                        </div>
                        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${task.progress}%` }}
                          ></div>
                        </div>
                      </div>
                    )}
                    {task.error && (
                      <div className="mt-2 p-2 bg-red-50 dark:bg-red-900/30 rounded-lg">
                        <p className="text-sm text-red-600 dark:text-red-400">{task.error}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === "files" && (
          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
            <div className="divide-y divide-slate-100 dark:divide-slate-700">
              {files.map((file) => (
                <div key={file.id} className="p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      {getStatusIcon(file.status)}
                      <div>
                        <h3 className="font-semibold text-slate-900 dark:text-slate-100">{file.name}</h3>
                        <div className="flex items-center gap-4 mt-1 text-sm text-slate-500 dark:text-slate-400">
                          <span>{file.type}</span>
                          <span>{formatFileSize(file.size)}</span>
                          <span>{new Date(file.uploadTime).toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(file.status)}`}>
                        {file.status}
                      </span>
                      <div className="mt-2 text-sm text-slate-500 dark:text-slate-400">
                        {file.tokensGenerated > 0 && <span>{file.tokensGenerated} tokens generated</span>}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === "tokens" && (
          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
            <div className="divide-y divide-slate-100 dark:divide-slate-700">
              {tokenUsage.map((usage, index) => (
                <div key={index} className="p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-slate-900 dark:text-slate-100">{usage.model}</h3>
                      <div className="flex items-center gap-4 mt-1 text-sm text-slate-500 dark:text-slate-400">
                        <span>{usage.requests} requests</span>
                        <span>Last used: {new Date(usage.lastUsed).toLocaleString()}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-slate-900 dark:text-slate-100">
                        {usage.totalTokens.toLocaleString()}
                      </div>
                      <div className="text-sm text-slate-500 dark:text-slate-400">
                        ${usage.cost.toFixed(4)}
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-slate-500 dark:text-slate-400">Input: </span>
                      <span className="text-slate-900 dark:text-slate-100">{usage.inputTokens.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 dark:text-slate-400">Output: </span>
                      <span className="text-slate-900 dark:text-slate-100">{usage.outputTokens.toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === "system" && systemMetrics && (
          <div className="space-y-6">
            {/* System Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-3">
                  <Cpu className="w-8 h-8 text-blue-500" />
                  <div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">CPU Usage</p>
                    <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{systemMetrics.cpuUsage}%</p>
                  </div>
                </div>
                <div className="mt-3 w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${systemMetrics.cpuUsage}%` }}
                  ></div>
                </div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-3">
                  <Database className="w-8 h-8 text-green-500" />
                  <div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Memory Usage</p>
                    <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{systemMetrics.memoryUsage}%</p>
                  </div>
                </div>
                <div className="mt-3 w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${systemMetrics.memoryUsage}%` }}
                  ></div>
                </div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-3">
                  <HardDrive className="w-8 h-8 text-purple-500" />
                  <div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Disk Usage</p>
                    <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{systemMetrics.diskUsage}%</p>
                  </div>
                </div>
                <div className="mt-3 w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${systemMetrics.diskUsage}%` }}
                  ></div>
                </div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-3">
                  <Activity className="w-8 h-8 text-amber-500" />
                  <div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Connections</p>
                    <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">{systemMetrics.activeConnections}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* System Info */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">System Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">System Uptime</p>
                  <p className="text-xl font-semibold text-slate-900 dark:text-slate-100">{systemMetrics.uptime}</p>
                </div>
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Last Updated</p>
                  <p className="text-xl font-semibold text-slate-900 dark:text-slate-100">
                    {new Date(systemMetrics.lastUpdated).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}