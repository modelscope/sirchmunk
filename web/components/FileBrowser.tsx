"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Folder,
  FileText,
  ChevronRight,
  ArrowUp,
  Home,
  Loader2,
  HardDrive,
  X,
} from "lucide-react";
import { apiUrl } from "@/lib/api";

interface FileItem {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
  modified: number;
}

interface FileBrowserProps {
  mode: "files" | "directory";
  onSelect: (path: string) => void;
  onCancel: () => void;
  t: (key: string) => string;
}

function formatSize(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export default function FileBrowser({ mode, onSelect, onCancel, t }: FileBrowserProps) {
  const [currentPath, setCurrentPath] = useState("/");
  const [parentPath, setParentPath] = useState("/");
  const [items, setItems] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pathInput, setPathInput] = useState("/");
  const [showHidden, setShowHidden] = useState(false);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);

  const browse = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    setSelectedItem(null);
    try {
      const res = await fetch(
        apiUrl(`/api/v1/file-browser?path=${encodeURIComponent(path)}&show_hidden=${showHidden}`)
      );
      const result = await res.json();
      if (result.success) {
        setCurrentPath(result.data.current_path);
        setParentPath(result.data.parent_path);
        setItems(result.data.items);
        setPathInput(result.data.current_path);
      } else {
        setError(result.error || "Failed to browse");
      }
    } catch (err) {
      setError("Failed to connect to server");
    } finally {
      setLoading(false);
    }
  }, [showHidden]);

  useEffect(() => {
    browse(currentPath);
  }, [showHidden]);

  useEffect(() => {
    browse("/");
  }, []);

  const handleItemClick = (item: FileItem) => {
    if (item.is_dir) {
      browse(item.path);
    } else if (mode === "files") {
      setSelectedItem(item.path);
    }
  };

  const handleItemDoubleClick = (item: FileItem) => {
    if (item.is_dir && mode === "directory") {
      onSelect(item.path);
    } else if (!item.is_dir && mode === "files") {
      onSelect(item.path);
    }
  };

  const handleConfirm = () => {
    if (mode === "directory") {
      onSelect(currentPath);
    } else if (selectedItem) {
      onSelect(selectedItem);
    }
  };

  const handlePathInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      browse(pathInput);
    }
  };

  const pathParts = currentPath.split("/").filter(Boolean);

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[99999]">
      <div className="bg-white dark:bg-slate-800 rounded-2xl w-[600px] max-h-[80vh] mx-4 shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {mode === "directory" ? t("Select Folder") : t("Select File")}
            </h3>
          </div>
          <button
            onClick={onCancel}
            className="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Path Bar */}
        <div className="px-4 py-2 border-b border-slate-200 dark:border-slate-700 flex items-center gap-2">
          <button
            onClick={() => browse("/")}
            className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400 transition-colors shrink-0"
            title="Root"
          >
            <Home className="w-4 h-4" />
          </button>
          <button
            onClick={() => browse(parentPath)}
            disabled={currentPath === parentPath}
            className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400 disabled:opacity-30 transition-colors shrink-0"
            title={t("Go up")}
          >
            <ArrowUp className="w-4 h-4" />
          </button>
          <input
            type="text"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={handlePathInputKeyDown}
            onBlur={() => setPathInput(currentPath)}
            className="flex-1 px-2 py-1 text-sm bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        {/* Breadcrumb */}
        <div className="px-4 py-1.5 flex items-center gap-0.5 text-xs text-slate-500 dark:text-slate-400 overflow-x-auto flex-shrink-0">
          <button onClick={() => browse("/")} className="hover:text-blue-600 dark:hover:text-blue-400 shrink-0">/</button>
          {pathParts.map((part, i) => {
            const fullPath = "/" + pathParts.slice(0, i + 1).join("/");
            return (
              <span key={i} className="flex items-center shrink-0">
                <ChevronRight className="w-3 h-3 mx-0.5" />
                <button
                  onClick={() => browse(fullPath)}
                  className="hover:text-blue-600 dark:hover:text-blue-400 truncate max-w-[120px]"
                >
                  {part}
                </button>
              </span>
            );
          })}
        </div>

        {/* File List */}
        <div className="flex-1 overflow-y-auto min-h-0 px-2 py-1">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-slate-500 dark:text-slate-400">
              <Loader2 className="w-5 h-5 animate-spin mr-2" />
              <span>{t("Loading")}...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-12 text-red-500 dark:text-red-400 text-sm">
              {error}
            </div>
          ) : items.length === 0 ? (
            <div className="flex items-center justify-center py-12 text-slate-400 dark:text-slate-500 text-sm">
              {t("Empty folder")}
            </div>
          ) : (
            <div className="space-y-0.5">
              {items.map((item) => {
                const isSelected = selectedItem === item.path;
                const isSelectable = item.is_dir || mode === "files";
                return (
                  <button
                    key={item.path}
                    onClick={() => handleItemClick(item)}
                    onDoubleClick={() => handleItemDoubleClick(item)}
                    className={`w-full text-left px-3 py-2 rounded-lg flex items-center gap-3 transition-colors ${
                      isSelected
                        ? "bg-blue-100 dark:bg-blue-900/40 border border-blue-300 dark:border-blue-700"
                        : "hover:bg-slate-50 dark:hover:bg-slate-700/50 border border-transparent"
                    } ${!isSelectable ? "opacity-40" : ""}`}
                    disabled={!isSelectable}
                  >
                    {item.is_dir ? (
                      <Folder className="w-5 h-5 text-amber-500 dark:text-amber-400 shrink-0" />
                    ) : (
                      <FileText className="w-5 h-5 text-slate-400 dark:text-slate-500 shrink-0" />
                    )}
                    <span className="flex-1 text-sm text-slate-800 dark:text-slate-200 truncate">
                      {item.name}
                    </span>
                    {!item.is_dir && item.size !== null && (
                      <span className="text-xs text-slate-400 dark:text-slate-500 shrink-0">
                        {formatSize(item.size)}
                      </span>
                    )}
                    {item.is_dir && (
                      <ChevronRight className="w-4 h-4 text-slate-300 dark:text-slate-600 shrink-0" />
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-200 dark:border-slate-700 px-5 py-3 flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showHidden}
              onChange={(e) => setShowHidden(e.target.checked)}
              className="rounded border-slate-300 dark:border-slate-600"
            />
            {t("Show hidden files")}
          </label>
          <div className="flex gap-2">
            <button
              onClick={onCancel}
              className="px-4 py-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 border border-slate-200 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
            >
              {t("Cancel")}
            </button>
            <button
              onClick={handleConfirm}
              disabled={mode === "files" && !selectedItem}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {mode === "directory"
                ? t("Select Current Folder")
                : t("Select")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
