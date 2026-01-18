"use client";

import { useState, useEffect } from "react";
import {
  Monitor,
  Activity,
  Database,
  MessageSquare,
  Brain,
  Server,
  Cpu,
  HardDrive,
  Wifi,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Loader2,
  HelpCircle,
} from "lucide-react";
import { apiUrl } from "@/lib/api";
import { getTranslation, type Language } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";

// === Types ===

interface SystemMetrics {
  cpu: {
    usage_percent: number;
    count: number;
    process_percent: number;
  };
  memory: {
    usage_percent: number;
    total_gb: number;
    used_gb: number;
    available_gb: number;
    process_mb: number;
  };
  disk: {
    usage_percent: number;
    total_gb: number;
    used_gb: number;
    free_gb: number;
  };
  network: {
    active_connections: number;
  };
  uptime: string;
  timestamp: string;
}

interface ChatActivity {
  total_sessions: number;
  total_messages: number;
  active_sessions: number;
  recent_sessions: Array<{
    session_id: string;
    title: string;
    message_count: number;
    created_at: number;
    updated_at: number;
  }>;
  time_window_hours: number;
}

interface KnowledgeActivity {
  total_clusters: number;
  average_confidence: number;
  recent_clusters: Array<{
    id: string;
    name: string;
    lifecycle: string;
    last_modified: string;
    confidence: number;
  }>;
  lifecycle_distribution: Record<string, number>;
}

interface StorageInfo {
  work_path: string;
  cache_path: string;
  databases: Record<string, {
    path: string;
    size_mb: number;
    exists: boolean;
  }>;
  total_cache_size_mb: number;
}

interface HealthStatus {
  overall_status: string;
  status_color: string;
  health_score: number;
  issues: string[];
  services: Record<string, {
    status: string;
    healthy: boolean;
  }>;
  timestamp: string;
}

interface MonitorOverview {
  system: SystemMetrics;
  chat: ChatActivity;
  knowledge: KnowledgeActivity;
  storage: StorageInfo;
  health: HealthStatus;
  timestamp: string;
}

export default function MonitorPage() {
  const { uiSettings } = useGlobal();
  const t = (key: string) => getTranslation((uiSettings?.language || "en") as Language, key);

  const [overview, setOverview] = useState<MonitorOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Fetch overview data
  const fetchOverview = async () => {
    try {
      setRefreshing(true);
      setError("");

      const response = await fetch(apiUrl("/api/v1/monitor/overview"));
      if (!response.ok) throw new Error("Failed to fetch monitoring data");

      const result = await response.json();
      if (result.success && result.data) {
        setOverview(result.data);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchOverview();
  }, []);

  // Auto-refresh every 5 seconds
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchOverview();
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "excellent":
        return "text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30";
      case "good":
        return "text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30";
      case "warning":
        return "text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30";
      case "critical":
        return "text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30";
      default:
        return "text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-700";
    }
  };

  // Get progress bar color
  const getProgressColor = (percent: number) => {
    if (percent >= 90) return "bg-red-500";
    if (percent >= 75) return "bg-yellow-500";
    if (percent >= 50) return "bg-blue-500";
    return "bg-green-500";
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-slate-600 dark:text-slate-400">{t("Loading monitoring data...")}</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
        <div className="text-center max-w-md">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">{t("Error")}</h2>
          <p className="text-slate-600 dark:text-slate-400">{error}</p>
          <button
            onClick={() => { setLoading(true); fetchOverview(); }}
            className="mt-4 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg"
          >
            {t("Retry")}
          </button>
        </div>
      </div>
    );
  }

  if (!overview) return null;

  const { system, chat, knowledge, storage, health } = overview;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Monitor className="w-7 h-7 text-blue-500 dark:text-blue-400" />
              <div>
                <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {t("System Monitor")}
                </h1>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                  {t("Real-time system monitoring and metrics")}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Auto-refresh toggle */}
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                  autoRefresh
                    ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                    : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400"
                }`}
              >
                <Activity className="w-4 h-4" />
                <span className="text-sm font-medium">
                  {autoRefresh ? t("Auto-refresh ON") : t("Auto-refresh OFF")}
                </span>
              </button>

              {/* Manual refresh button */}
              <button
                onClick={fetchOverview}
                disabled={refreshing}
                className="flex items-center gap-2 px-3 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-300 dark:disabled:bg-slate-700 text-white rounded-lg transition-colors"
              >
                <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                <span className="text-sm font-medium">{t("Refresh")}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 py-6 space-y-6">
        {/* Health Status Banner */}
        <div className={`rounded-xl border-2 p-6 ${getStatusColor(health.overall_status)}`}>
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              {health.overall_status === "excellent" || health.overall_status === "good" ? (
                <CheckCircle className="w-10 h-10 flex-shrink-0" />
              ) : (
                <AlertCircle className="w-10 h-10 flex-shrink-0" />
              )}
              <div>
                <h2 className="text-xl font-bold capitalize mb-1">
                  {health.overall_status} Health
                </h2>
                <p className="text-sm opacity-90 mb-2">
                  System is operating at {health.health_score}% capacity
                </p>
                {health.issues.length > 0 && (
                  <ul className="text-sm space-y-1">
                    {health.issues.map((issue, i) => (
                      <li key={i}>• {issue}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold">{health.health_score}</div>
              <div className="text-xs opacity-75">Health Score</div>
            </div>
          </div>
        </div>

        {/* System Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* CPU Usage */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("CPU Usage")}
                </p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {system.cpu.usage_percent}%
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                <Cpu className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
            <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${getProgressColor(system.cpu.usage_percent)}`}
                style={{ width: `${Math.min(system.cpu.usage_percent, 100)}%` }}
              />
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
              {system.cpu.count} cores • Process: {system.cpu.process_percent}%
            </p>
          </div>

          {/* Memory Usage */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("Memory")}
                </p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {system.memory.usage_percent}%
                </p>
              </div>
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <Activity className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
            <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${getProgressColor(system.memory.usage_percent)}`}
                style={{ width: `${Math.min(system.memory.usage_percent, 100)}%` }}
              />
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
              {system.memory.used_gb.toFixed(1)} / {system.memory.total_gb.toFixed(1)} GB
            </p>
          </div>

          {/* Disk Usage */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("Disk")}
                </p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {system.disk.usage_percent}%
                </p>
              </div>
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <HardDrive className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
            <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${getProgressColor(system.disk.usage_percent)}`}
                style={{ width: `${Math.min(system.disk.usage_percent, 100)}%` }}
              />
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
              {system.disk.free_gb.toFixed(1)} GB free of {system.disk.total_gb.toFixed(1)} GB
            </p>
          </div>

          {/* Uptime */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("Uptime")}
                </p>
                <p className="text-lg font-bold text-slate-900 dark:text-slate-100">
                  {system.uptime}
                </p>
              </div>
              <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                <Clock className="w-6 h-6 text-orange-600 dark:text-orange-400" />
              </div>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
              {system.network.active_connections} active connections
            </p>
          </div>
        </div>

        {/* Activity Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chat Activity */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <MessageSquare className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Chat Activity")}
              </h3>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {chat.total_sessions}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">Total Sessions</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {chat.active_sessions}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">Active (24h)</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {chat.total_messages}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">Messages</div>
              </div>
            </div>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {chat.recent_sessions.length > 0 ? (
                chat.recent_sessions.map((session) => (
                  <div
                    key={session.session_id}
                    className="p-3 rounded-lg bg-slate-50 dark:bg-slate-700/50"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                          {session.title}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          {session.message_count} messages •{" "}
                          {new Date(session.updated_at * 1000).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-slate-500 dark:text-slate-400 text-center py-4">
                  No recent sessions
                </p>
              )}
            </div>
          </div>

          {/* Knowledge Activity */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-5 h-5 text-purple-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Knowledge Clusters")}
              </h3>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {knowledge.total_clusters}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">Total Clusters</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {(knowledge.average_confidence * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">Avg Confidence</div>
              </div>
            </div>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {knowledge.recent_clusters.length > 0 ? (
                knowledge.recent_clusters.map((cluster) => (
                  <div
                    key={cluster.id}
                    className="p-3 rounded-lg bg-slate-50 dark:bg-slate-700/50"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                          {cluster.name}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          {cluster.lifecycle} •{" "}
                          {cluster.confidence ? `${(cluster.confidence * 100).toFixed(0)}%` : "N/A"}
                        </p>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-slate-500 dark:text-slate-400 text-center py-4">
                  No clusters created yet
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Storage and Services */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Storage Info */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Database className="w-5 h-5 text-green-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Storage")}
              </h3>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Total Cache</span>
                  <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    {storage.total_cache_size_mb.toFixed(2)} MB
                  </span>
                </div>
              </div>

              {Object.entries(storage.databases).map(([name, db]) => (
                <div key={name}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400 capitalize">
                      {name}
                    </span>
                    <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {db.size_mb.toFixed(2)} MB
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Services Status */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Server className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Services")}
              </h3>
            </div>

            <div className="space-y-3">
              {Object.entries(health.services).map(([name, service]) => (
                <div key={name} className="flex items-center justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400 capitalize">
                    {name.replace(/_/g, " ")}
                  </span>
                  <div className="flex items-center gap-2">
                    {service.healthy ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-red-500" />
                    )}
                    <span className="text-sm font-medium text-slate-900 dark:text-slate-100">
                      {service.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
