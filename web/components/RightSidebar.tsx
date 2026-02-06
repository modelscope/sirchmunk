"use client";

import { useState, useEffect, useRef } from "react";
import {
  FileText,
  Monitor,
  Video,
  Image,
  FileSpreadsheet,
  RefreshCw,
  Cloud,
  FolderOpen,
  RotateCw,
  Settings,
  ChevronDown,
  ChevronRight,
  Loader2,
  CheckCircle,
  AlertCircle,
  Clock,
  ExternalLink,
  ChevronsRight,
  ChevronsLeft,
  GripVertical,
} from "lucide-react";
import { apiUrl } from "@/lib/api";
import { getTranslation, type Language } from "@/lib/i18n";
import { useGlobal } from "@/context/GlobalContext";

interface HubFile {
  id: string;
  name: string;
  type: string;
  size: string;
  modified: string;
  status: "synced" | "pending" | "error";
  url?: string;
}

interface QuickAction {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  description: string;
  action: () => void;
  loading?: boolean;
}

interface RightSidebarProps {
  isCollapsed?: boolean;
  onToggle?: () => void;
  width?: number;
  onWidthChange?: (width: number) => void;
}

export default function RightSidebar({
  isCollapsed = false,
  onToggle,
  width = 384, // 默认宽度为原来的0.8倍 (480 * 0.8 = 384)
  onWidthChange
}: RightSidebarProps) {
  const { uiSettings } = useGlobal();
  const t = (key: string) => getTranslation(uiSettings.language as Language, key);

  const [hubFiles, setHubFiles] = useState<HubFile[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    quickActions: true,
    hubFiles: true,
  });

  // Resizing state
  const [isResizing, setIsResizing] = useState(false);
  const resizeRef = useRef<HTMLDivElement>(null);

  // Quick Actions
  const [actionStates, setActionStates] = useState<Record<string, boolean>>({});

  const quickActions: QuickAction[] = [
    {
      id: "export-pdf",
      label: t("Export PDF"),
      icon: FileText,
      color: "red",
      description: "Export conversation to PDF",
      action: () => handleQuickAction("export-pdf"),
      loading: actionStates["export-pdf"],
    },
    {
      id: "generate-ppt",
      label: t("Generate PPT"),
      icon: Monitor,
      color: "orange",
      description: "Create presentation slides",
      action: () => handleQuickAction("generate-ppt"),
      loading: actionStates["generate-ppt"],
    },
    {
      id: "convert-doc",
      label: t("Convert Document"),
      icon: FileSpreadsheet,
      color: "green",
      description: "Convert file formats",
      action: () => handleQuickAction("convert-doc"),
      loading: actionStates["convert-doc"],
    },
    {
      id: "generate-video",
      label: t("Generate Video"),
      icon: Video,
      color: "purple",
      description: "Create video content",
      action: () => handleQuickAction("generate-video"),
      loading: actionStates["generate-video"],
    },
    {
      id: "create-image",
      label: t("Create Image"),
      icon: Image,
      color: "blue",
      description: "Generate images",
      action: () => handleQuickAction("create-image"),
      loading: actionStates["create-image"],
    },
    {
      id: "export-excel",
      label: t("Export Excel"),
      icon: FileSpreadsheet,
      color: "emerald",
      description: "Export data to Excel",
      action: () => handleQuickAction("export-excel"),
      loading: actionStates["export-excel"],
    },
  ];

  // Fetch hub files
  const fetchHubFiles = async () => {
    setIsLoadingFiles(true);
    try {
      const response = await fetch(apiUrl("/api/v1/hub/files"));
      const result = await response.json();
      if (result.success) {
        setHubFiles(result.data);
      }
    } catch (error) {
      console.error("Failed to fetch hub files:", error);
    } finally {
      setIsLoadingFiles(false);
    }
  };

  // Sync hub files
  const syncHubFiles = async () => {
    setIsSyncing(true);
    try {
      const response = await fetch(apiUrl("/api/v1/hub/sync"), {
        method: "POST",
      });
      const result = await response.json();
      if (result.success) {
        await fetchHubFiles(); // Refresh files after sync
      }
    } catch (error) {
      console.error("Failed to sync hub files:", error);
    } finally {
      setIsSyncing(false);
    }
  };

  // Handle quick actions
  const handleQuickAction = async (actionId: string) => {
    setActionStates(prev => ({ ...prev, [actionId]: true }));
    
    try {
      const response = await fetch(apiUrl(`/api/v1/tools/${actionId}`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          // Add any necessary data based on current context
          timestamp: new Date().toISOString(),
        }),
      });
      
      const result = await response.json();
      if (result.success) {
        // Handle success (e.g., download file, show notification)
        if (result.data?.download_url) {
          window.open(result.data.download_url, "_blank");
        }
      }
    } catch (error) {
      console.error(`Failed to execute ${actionId}:`, error);
    } finally {
      setActionStates(prev => ({ ...prev, [actionId]: false }));
    }
  };

  // Toggle section expansion
  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  // Get file status icon
  const getStatusIcon = (status: HubFile["status"]) => {
    switch (status) {
      case "synced":
        return <CheckCircle className="w-3 h-3 text-green-500" />;
      case "pending":
        return <Clock className="w-3 h-3 text-yellow-500" />;
      case "error":
        return <AlertCircle className="w-3 h-3 text-red-500" />;
      default:
        return null;
    }
  };

  // Get file type color
  const getFileTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      pdf: "text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400",
      docx: "text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400",
      pptx: "text-orange-600 bg-orange-100 dark:bg-orange-900/30 dark:text-orange-400",
      xlsx: "text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400",
      txt: "text-slate-600 bg-slate-100 dark:bg-slate-700 dark:text-slate-400",
      default: "text-slate-600 bg-slate-100 dark:bg-slate-700 dark:text-slate-400",
    };
    return colors[type.toLowerCase()] || colors.default;
  };

  // Handle mouse resize
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !onWidthChange) return;

      const newWidth = window.innerWidth - e.clientX;
      const minWidth = 280; // 最小宽度
      const maxWidth = 600; // 最大宽度

      const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      onWidthChange(clampedWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing, onWidthChange]);

  // Load files on component mount
  useEffect(() => {
    fetchHubFiles();
  }, []);

  // Collapsed sidebar
  if (isCollapsed) {
    return (
      <div className="w-16 h-full bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-700 flex flex-col">
        {/* Collapse toggle button */}
        <div className="flex-shrink-0 p-2 border-b border-slate-200 dark:border-slate-700">
          <button
            onClick={onToggle}
            className="w-full flex items-center justify-center p-2 rounded-md text-slate-400 dark:text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            title={t("Expand sidebar")}
          >
            <ChevronsLeft className="w-4 h-4" />
          </button>
        </div>

        {/* Collapsed quick actions */}
        <div className="flex-1 p-2 space-y-2">
          {quickActions.slice(0, 4).map((action) => (
            <button
              key={action.id}
              onClick={action.action}
              disabled={action.loading}
              className="w-full p-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
              title={action.label}
            >
              {action.loading ? (
                <Loader2 className="w-4 h-4 animate-spin text-slate-500 mx-auto" />
              ) : (
                <action.icon className={`w-4 h-4 mx-auto text-${action.color}-600 dark:text-${action.color}-400`} />
              )}
            </button>
          ))}
        </div>

        {/* Collapsed hub indicator */}
        <div className="flex-shrink-0 p-2 border-t border-slate-200 dark:border-slate-700">
          <div className="w-full p-2 rounded-lg bg-slate-50 dark:bg-slate-800 flex items-center justify-center">
            <Cloud className="w-4 h-4 text-slate-400" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Resize Handle */}
      <div
        ref={resizeRef}
        className={`absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500 transition-colors z-10 ${
          isResizing ? 'bg-blue-500' : 'bg-transparent hover:bg-slate-300 dark:hover:bg-slate-600'
        }`}
        onMouseDown={handleMouseDown}
      >
        <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-0 hover:opacity-100 transition-opacity">
          <GripVertical className="w-3 h-3 text-slate-400" />
        </div>
      </div>

      <div
        className="h-full bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-700 flex flex-col"
        style={{ width: `${width}px` }}
      >
        {/* Header with collapse button */}
        <div className="flex-shrink-0 px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
          {t("Tools & Hub")}
        </h2>
        <button
          onClick={onToggle}
          className="text-slate-400 hover:text-blue-500 dark:hover:text-blue-400 p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
          title={t("Collapse sidebar")}
        >
          <ChevronsRight className="w-4 h-4" />
        </button>
      </div>

      {/* Quick Actions Section */}
      <div className="flex-shrink-0 border-b border-slate-200 dark:border-slate-700">
        <button
          onClick={() => toggleSection("quickActions")}
          className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            <span>{t("Quick Actions")}</span>
          </div>
          {expandedSections.quickActions ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>

        {expandedSections.quickActions && (
          <div className="px-4 pb-4">
            <div className="grid grid-cols-3 gap-3">
              {quickActions.map((action) => (
                <button
                  key={action.id}
                  onClick={action.action}
                  disabled={action.loading}
                  className={`group p-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:shadow-md hover:border-${action.color}-300 dark:hover:border-${action.color}-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <div
                    className={`w-8 h-8 rounded-lg bg-${action.color}-100 dark:bg-${action.color}-900/30 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform`}
                  >
                    {action.loading ? (
                      <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
                    ) : (
                      <action.icon
                        className={`w-4 h-4 text-${action.color}-600 dark:text-${action.color}-400`}
                      />
                    )}
                  </div>
                  <h4 className="font-medium text-slate-900 dark:text-slate-100 text-xs mb-1 leading-tight">
                    {action.label}
                  </h4>
                  <p className="text-xs text-slate-500 dark:text-slate-400 leading-tight">
                    {action.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Hub Files Section */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex-shrink-0 border-b border-slate-200 dark:border-slate-700">
          <button
            onClick={() => toggleSection("hubFiles")}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Cloud className="w-4 h-4" />
              <span>{t("Hub Files")}</span>
            </div>
            {expandedSections.hubFiles ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>

          {expandedSections.hubFiles && (
            <div className="px-4 pb-3">
              <div className="flex items-center gap-2 mb-3">
                <button
                  onClick={fetchHubFiles}
                  disabled={isLoadingFiles}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={`w-3 h-3 ${isLoadingFiles ? "animate-spin" : ""}`} />
                  {t("Refresh")}
                </button>
                <button
                  onClick={syncHubFiles}
                  disabled={isSyncing}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/70 transition-colors disabled:opacity-50"
                >
                  <RotateCw className={`w-3 h-3 ${isSyncing ? "animate-spin" : ""}`} />
                  {t("Sync")}
                </button>
              </div>
            </div>
          )}
        </div>

        {expandedSections.hubFiles && (
          <div className="flex-1 overflow-y-auto">
            {isLoadingFiles ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 animate-spin text-slate-400" />
              </div>
            ) : hubFiles.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
                <FolderOpen className="w-8 h-8 text-slate-400 mb-2" />
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">
                  {t("No files found")}
                </p>
                <p className="text-xs text-slate-400 dark:text-slate-500">
                  {t("Sync to load files from hub")}
                </p>
              </div>
            ) : (
              <div className="px-3 py-2 space-y-2">
                {hubFiles.map((file) => (
                  <div
                    key={file.id}
                    className="group p-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:shadow-sm transition-all"
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-medium ${getFileTypeColor(
                          file.type
                        )}`}
                      >
                        {file.type.toUpperCase()}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-medium text-slate-900 dark:text-slate-100 text-sm truncate">
                            {file.name}
                          </h4>
                          {getStatusIcon(file.status)}
                        </div>
                        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                          <span>{file.size}</span>
                          <span>{file.modified}</span>
                        </div>
                      </div>
                      {file.url && (
                        <button
                          onClick={() => window.open(file.url, "_blank")}
                          className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-all"
                        >
                          <ExternalLink className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      </div>
    </div>
  );
}