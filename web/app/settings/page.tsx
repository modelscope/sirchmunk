"use client";

import { useState, useEffect } from "react";
import {
  Settings as SettingsIcon,
  Sun,
  Moon,
  Globe,
  Save,
  Loader2,
  Check,
  AlertCircle,
  Key,
} from "lucide-react";
import { apiUrl } from "@/lib/api";
import { getTranslation, type Language } from "@/lib/i18n";
import { setTheme } from "@/lib/theme";
import { useGlobal } from "@/context/GlobalContext";

// === Types ===

interface UISettings {
  theme: "light" | "dark";
  language: "zh" | "en";
}

interface EnvironmentVariable {
  value: string;
  default: string;
  description: string;
  category: string;
  sensitive?: boolean;
}

interface SettingsData {
  ui: UISettings;
  environment: Record<string, EnvironmentVariable>;
}

type SettingsTab = "interface" | "environment";

export default function SettingsPage() {
  const { uiSettings, refreshSettings } = useGlobal();
  const t = (key: string) => getTranslation((uiSettings?.language || "en") as Language, key);
  
  const [activeTab, setActiveTab] = useState<SettingsTab>("interface");
  const [data, setData] = useState<SettingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [error, setError] = useState("");

  // Edit states
  const [editedUI, setEditedUI] = useState<UISettings | null>(null);
  const [editedEnv, setEditedEnv] = useState<Record<string, string>>({});

  // Fetch settings
  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      setLoading(true);
      setError("");
      
      const response = await fetch(apiUrl("/api/v1/settings"));
      if (!response.ok) throw new Error("Failed to fetch settings");
      
      const result = await response.json();
      if (result.success && result.data) {
        setData(result.data);
        setEditedUI(result.data.ui);
        
        // Initialize edited env with current values
        const envValues: Record<string, string> = {};
        Object.entries(result.data.environment || {}).forEach(([key, info]: [string, any]) => {
          if (key === "WORK_PATH") return;
          envValues[key] = info.value || "";
        });
        setEditedEnv(envValues);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSettings = async () => {
    try {
    setSaving(true);
    setError("");
      setSaveSuccess(false);

      const payload: any = {};
      
      // Always include UI settings
      if (editedUI) {
        payload.ui = editedUI;
      }
      
      // Include environment variables if edited
      if (Object.keys(editedEnv).length > 0) {
        const envPayload = { ...editedEnv };
        delete envPayload.WORK_PATH;
        payload.environment = envPayload;
      }

      const response = await fetch(apiUrl("/api/v1/settings"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Failed to save settings");
      
      const result = await response.json();
      if (result.success) {
        setSaveSuccess(true);
        
        // Update theme immediately if changed
        if (editedUI?.theme && editedUI.theme !== data?.ui?.theme) {
        setTheme(editedUI.theme);
      }

        // Refresh settings in global context
      await refreshSettings();

        // Reload settings to get updated values
        await fetchSettings();
        
        setTimeout(() => setSaveSuccess(false), 3000);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-slate-600 dark:text-slate-400">{t("Loading settings...")}</p>
        </div>
      </div>
    );
  }

    return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-50 to-blue-50/30 dark:from-slate-900 dark:to-blue-950/20">
      {/* Header */}
      <div className="flex-none border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <SettingsIcon className="w-6 h-6 text-blue-500 dark:text-blue-400" />
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {t("System Settings")}
            </h1>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                  {t("Manage system configuration and preferences")}
                </p>
          </div>
            </div>
            
          <button
              onClick={handleSaveSettings}
            disabled={saving}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-300 dark:disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors"
          >
            {saving ? (
                <>
              <Loader2 className="w-4 h-4 animate-spin" />
                  {t("Saving...")}
                </>
            ) : saveSuccess ? (
                <>
              <Check className="w-4 h-4" />
                  {t("Saved!")}
                </>
            ) : (
                <>
              <Save className="w-4 h-4" />
                  {t("Save All Changes")}
                </>
            )}
          </button>
        </div>
      </div>

        {/* Tabs */}
        <div className="px-6 flex gap-1">
          <button
            onClick={() => setActiveTab("interface")}
            className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
              activeTab === "interface"
                ? "border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
            }`}
          >
            <div className="flex items-center gap-2">
              <Globe className="w-4 h-4" />
              {t("Interface Preferences")}
            </div>
          </button>
          
          <button
            onClick={() => setActiveTab("environment")}
            className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
              activeTab === "environment"
                ? "border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
            }`}
          >
            <div className="flex items-center gap-2">
            <Key className="w-4 h-4" />
            {t("Environment Variables")}
        </div>
                </button>
              </div>
              </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Error Message */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium text-red-900 dark:text-red-200">{t("Error")}</p>
                <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
                  </div>
                  </div>
          )}

          {/* Interface Preferences */}
          {activeTab === "interface" && editedUI && (
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
              <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-700">
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                    {t("Interface Preferences")}
                </h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                  {t("Customize your interface appearance and language")}
                </p>
                </div>
              
              <div className="p-6 space-y-6">
                {/* Theme */}
                  <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                      {t("Theme")}
                    </label>
                  <div className="grid grid-cols-2 gap-3">
                        <button
                      onClick={() => setEditedUI({ ...editedUI, theme: "light" })}
                      className={`flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
                        editedUI.theme === "light"
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                          : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
                      }`}
                    >
                      <Sun className={`w-5 h-5 ${
                        editedUI.theme === "light" 
                          ? "text-blue-600 dark:text-blue-400" 
                          : "text-slate-400"
                      }`} />
                      <span className={`font-medium ${
                        editedUI.theme === "light" 
                          ? "text-blue-900 dark:text-blue-100" 
                          : "text-slate-600 dark:text-slate-400"
                      }`}>
                        {t("Light Mode")}
                          </span>
                        </button>
                    
                              <button
                      onClick={() => setEditedUI({ ...editedUI, theme: "dark" })}
                      className={`flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
                        editedUI.theme === "dark"
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                          : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
                      }`}
                    >
                      <Moon className={`w-5 h-5 ${
                        editedUI.theme === "dark" 
                              ? "text-blue-600 dark:text-blue-400"
                          : "text-slate-400"
                      }`} />
                      <span className={`font-medium ${
                        editedUI.theme === "dark" 
                          ? "text-blue-900 dark:text-blue-100" 
                          : "text-slate-600 dark:text-slate-400"
                      }`}>
                        {t("Dark Mode")}
                        </span>
                    </button>
                      </div>
                    </div>

                {/* Language */}
              <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                    {t("Language")}
                  </label>
                  <div className="grid grid-cols-2 gap-3">
              <button
                      onClick={() => setEditedUI({ ...editedUI, language: "en" })}
                      className={`flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
                        editedUI.language === "en"
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                          : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
                      }`}
                    >
                      <span className={`text-xl ${
                        editedUI.language === "en" ? "opacity-100" : "opacity-40"
                      }`}>🇺🇸</span>
                      <span className={`font-medium ${
                        editedUI.language === "en" 
                          ? "text-blue-900 dark:text-blue-100" 
                          : "text-slate-600 dark:text-slate-400"
                      }`}>
                        English
                      </span>
              </button>
                    
                  <button
                      onClick={() => setEditedUI({ ...editedUI, language: "zh" })}
                      className={`flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
                        editedUI.language === "zh"
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                          : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
                      }`}
                    >
                      <span className={`text-xl ${
                        editedUI.language === "zh" ? "opacity-100" : "opacity-40"
                      }`}>🇨🇳</span>
                      <span className={`font-medium ${
                        editedUI.language === "zh" 
                          ? "text-blue-900 dark:text-blue-100" 
                          : "text-slate-600 dark:text-slate-400"
                      }`}>
                        中文
                                </span>
                          </button>
                        </div>
                      </div>
                    </div>
              </div>
            )}

          {/* Environment Variables */}
          {activeTab === "environment" && data?.environment && (
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
              <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-700">
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                  {t("Environment Variables")}
                    </h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                  {t("Configure system and LLM environment variables")}
                      </p>
                    </div>

              <div className="p-6 space-y-4">
                {Object.entries(data.environment)
                  .filter(([key]) => key !== "WORK_PATH")
                  .map(([key, info]) => (
                  <div key={key} className="space-y-2">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                          {key}
                      </label>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                          {info.description}
                        </p>
                    </div>
                      <span className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400">
                        {info.category}
                      </span>
                    </div>

                      <input
                      type={info.sensitive ? "password" : "text"}
                      value={editedEnv[key] || ""}
                      onChange={(e) => setEditedEnv({ ...editedEnv, [key]: e.target.value })}
                      placeholder={`Default: ${info.default}`}
                      className="w-full px-3 py-2 border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                        </div>
                ))}
                            </div>
                          </div>
          )}
                      </div>
      </div>
    </div>
  );
}
