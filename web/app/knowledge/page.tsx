"use client";

import { useState, useEffect } from "react";
import {
  Database,
  TrendingUp,
  Activity,
  Brain,
  BarChart3,
  PieChart,
  Clock,
  Zap,
  Search,
  Loader2,
  AlertCircle,
  Check,
  ChevronRight,
} from "lucide-react";
import { apiUrl } from "@/lib/api";
import { getTranslation, type Language } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";

// === Types ===

interface LifecycleDistribution {
  STABLE: number;
  EMERGING: number;
  CONTESTED: number;
  DEPRECATED: number;
}

interface AbstractionLevelDistribution {
  TECHNIQUE: number;
  PRINCIPLE: number;
  PARADIGM: number;
  FOUNDATION: number;
  PHILOSOPHY: number;
}

interface Stats {
  min: number;
  max: number;
  avg: number;
}

interface ClusterSummary {
  id: string;
  name: string;
  confidence?: number;
  hotness?: number;
  last_modified: string;
}

interface TimelineData {
  date: string;
  count: number;
}

interface KnowledgeStats {
  overview: {
    total_clusters: number;
    avg_confidence: number;
  };
  lifecycle_distribution: LifecycleDistribution;
  abstraction_level_distribution: AbstractionLevelDistribution;
  confidence_stats: Stats;
  hotness_stats: Stats;
  recent_clusters: ClusterSummary[];
  top_confidence_clusters: ClusterSummary[];
  top_hotness_clusters: ClusterSummary[];
  timeline: TimelineData[];
}

interface Pattern {
  pattern: string;
  count: number;
}

export default function KnowledgePage() {
  const { uiSettings } = useGlobal();
  const t = (key: string) => getTranslation((uiSettings?.language || "en") as Language, key);

  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch stats
  useEffect(() => {
    fetchStats();
    fetchPatterns();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(apiUrl("/api/v1/knowledge/stats"));
      if (!response.ok) throw new Error("Failed to fetch stats");

      const result = await response.json();
      if (result.success && result.data) {
        setStats(result.data);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchPatterns = async () => {
    try {
      const response = await fetch(apiUrl("/api/v1/knowledge/patterns?limit=15"));
      if (!response.ok) return;

      const result = await response.json();
      if (result.success && result.data) {
        setPatterns(result.data);
      }
    } catch (err: any) {
      console.error("Failed to fetch patterns:", err);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    // TODO: Navigate to search results or show modal
    console.log("Search:", searchQuery);
  };

  // Lifecycle colors
  const lifecycleColors: Record<keyof LifecycleDistribution, string> = {
    STABLE: "bg-green-500",
    EMERGING: "bg-blue-500",
    CONTESTED: "bg-yellow-500",
    DEPRECATED: "bg-gray-500",
  };

  // Abstraction level colors
  const abstractionColors: Record<keyof AbstractionLevelDistribution, string> = {
    TECHNIQUE: "bg-purple-500",
    PRINCIPLE: "bg-indigo-500",
    PARADIGM: "bg-blue-500",
    FOUNDATION: "bg-cyan-500",
    PHILOSOPHY: "bg-teal-500",
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-slate-600 dark:text-slate-400">{t("Loading knowledge base...")}</p>
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
        </div>
      </div>
    );
  }

  if (!stats) {
    return null;
  }

  const total = stats.overview.total_clusters;
  const lifecycleData = stats.lifecycle_distribution;
  const abstractionData = stats.abstraction_level_distribution;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Database className="w-7 h-7 text-blue-500 dark:text-blue-400" />
        <div>
                <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {t("Knowledge Base Analytics")}
          </h1>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                  {t("Real-time insights and statistics")}
          </p>
        </div>
        </div>
      </div>

          {/* Search Bar (hidden for now) */}
          {false && (
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                placeholder={t("Search knowledge clusters...")}
                className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}
        </div>
                  </div>

      {/* Content */}
      <div className="px-6 py-6 space-y-6">
        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Clusters */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between">
                  <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("Total Clusters")}
                </p>
                <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                  {total.toLocaleString()}
                </p>
              </div>
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <Database className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
                  </div>
                </div>

          {/* Average Confidence */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("Avg Confidence")}
                </p>
                <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                  {(stats.overview.avg_confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
                </div>
              </div>

          {/* Max Hotness */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("Max Hotness")}
                </p>
                <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                  {(stats.hotness_stats.max * 100).toFixed(1)}%
                    </p>
                  </div>
              <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-orange-600 dark:text-orange-400" />
              </div>
            </div>
          </div>

          {/* Stable Clusters */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1">
                  {t("Stable Clusters")}
                </p>
                <p className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                  {lifecycleData.STABLE}
                    </p>
                  </div>
              <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg flex items-center justify-center">
                <Check className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
              </div>
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Lifecycle Distribution */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-6">
              <PieChart className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Lifecycle Distribution")}
              </h3>
                </div>

            <div className="space-y-4">
              {Object.entries(lifecycleData).map(([lifecycle, count]) => {
                const percentage = total > 0 ? (count / total) * 100 : 0;
                          return (
                  <div key={lifecycle}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {lifecycle}
                            </span>
                      <span className="text-sm text-slate-500 dark:text-slate-400">
                        {count} ({percentage.toFixed(1)}%)
                            </span>
                    </div>
                    <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2.5">
                      <div
                        className={`h-2.5 rounded-full ${lifecycleColors[lifecycle as keyof LifecycleDistribution]}`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Abstraction Level Distribution */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-6">
              <BarChart3 className="w-5 h-5 text-purple-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Abstraction Level Distribution")}
              </h3>
            </div>
            
            <div className="space-y-4">
              {Object.entries(abstractionData).map(([level, count]) => {
                const maxCount = Math.max(...Object.values(abstractionData));
                const percentage = maxCount > 0 ? (count / maxCount) * 100 : 0;
                      return (
                  <div key={level}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {level}
                      </span>
                      <span className="text-sm text-slate-500 dark:text-slate-400">
                        {count}
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2.5">
                      <div
                        className={`h-2.5 rounded-full ${abstractionColors[level as keyof AbstractionLevelDistribution]}`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
                          </div>
                            </div>
                            </div>

        {/* Stats Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Confidence Stats */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-6">
              <Activity className="w-5 h-5 text-green-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Confidence Statistics")}
              </h3>
                            </div>
            
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">{t("Min")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {(stats.confidence_stats.min * 100).toFixed(0)}%
                </p>
                        </div>
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">{t("Avg")}</p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {(stats.confidence_stats.avg * 100).toFixed(0)}%
                </p>
                        </div>
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">{t("Max")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {(stats.confidence_stats.max * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </div>

          {/* Hotness Stats */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-6">
              <Zap className="w-5 h-5 text-orange-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Hotness Statistics")}
              </h3>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">{t("Min")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {(stats.hotness_stats.min * 100).toFixed(0)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">{t("Avg")}</p>
                <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                  {(stats.hotness_stats.avg * 100).toFixed(0)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">{t("Max")}</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {(stats.hotness_stats.max * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Top Clusters */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Clusters */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Recent Activity")}
              </h3>
            </div>

            <div className="space-y-2">
              {stats.recent_clusters.slice(0, 10).map((cluster, index) => (
                <div
                  key={`recent-${cluster.id}-${index}`}
                  className="flex items-start gap-2 p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer"
                >
                  <ChevronRight className="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                      {cluster.name}
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {new Date(cluster.last_modified).toLocaleString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Confidence */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-green-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Top Confidence")}
              </h3>
              </div>

            <div className="space-y-2">
              {stats.top_confidence_clusters.slice(0, 10).map((cluster, index) => (
                <div
                  key={`confidence-${cluster.id}-${index}`}
                  className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer"
                >
                  <p className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate flex-1">
                    {cluster.name}
                  </p>
                  <span className="text-sm font-semibold text-green-600 dark:text-green-400 ml-2">
                    {cluster.confidence ? (cluster.confidence * 100).toFixed(0) : 0}%
                  </span>
                </div>
              ))}
              </div>
          </div>

          {/* Top Hotness */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Zap className="w-5 h-5 text-orange-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Hottest Topics")}
              </h3>
            </div>
            
            <div className="space-y-2">
              {stats.top_hotness_clusters.slice(0, 10).map((cluster, index) => (
                <div
                  key={`hotness-${cluster.id}-${index}`}
                  className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer"
                >
                  <p className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate flex-1">
                    {cluster.name}
                  </p>
                  <span className="text-sm font-semibold text-orange-600 dark:text-orange-400 ml-2">
                    {cluster.hotness ? (cluster.hotness * 100).toFixed(0) : 0}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Top Patterns */}
        {patterns.length > 0 && (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-5 h-5 text-purple-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {t("Common Patterns")}
              </h3>
              </div>

            <div className="flex flex-wrap gap-2">
              {patterns.map((pattern) => (
                <div
                  key={pattern.pattern}
                  className="px-3 py-1.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-sm font-medium hover:bg-purple-200 dark:hover:bg-purple-900/50 transition-colors cursor-pointer"
                >
                  {pattern.pattern} ({pattern.count})
                </div>
              ))}
              </div>
          </div>
        )}
        </div>
    </div>
  );
}
