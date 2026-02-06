/**
 * Type definitions for the question generation system.
 */

export interface QuestionState {
  global: {
    stage: "idle" | "planning" | "researching" | "generating" | "complete";
    startTime: number;
    totalQuestions: number;
    completedQuestions: number;
    failedQuestions: number;
    extendedQuestions: number;
  };
  planning: {
    topic: string;
    difficulty: string;
    questionType: string;
    queries: string[];
    progress: string;
  };
  tasks: Record<string, QuestionTask>;
  activeTaskIds: string[];
  results: QuestionResult[];
  failures: QuestionFailure[];
  subFocuses: QuestionFocus[];
  logs: QuestionLogEntry[];
}

export interface QuestionTask {
  id: string;
  focus: string;
  scenarioHint?: string;
  status: "pending" | "generating" | "validating" | "done" | "error";
  round?: number;
  lastUpdate: number;
  result?: QuestionResult;
  extended?: boolean;
  error?: string;
}

export interface QuestionLogEntry {
  id: string;
  timestamp: number;
  type: "info" | "success" | "warning" | "error" | "system";
  content: string;
}

export interface QuestionResult {
  success: boolean;
  question_id: string;
  question: {
    question: string;
    correct_answer: string;
    explanation: string;
  };
  validation: Record<string, unknown>;
  rounds: number;
  focus?: string | QuestionFocus;
  extended?: boolean;
}

export interface QuestionFailure {
  question_id: string;
  error: string;
  reason?: string;
}

export interface QuestionFocus {
  id: string;
  focus: string;
  scenario_hint?: string;
}

export interface QuestionEvent {
  type:
    | "progress"
    | "question_update"
    | "result"
    | "question_result"
    | "question_error"
    | "batch_summary"
    | "status"
    | "agent_status"
    | "token_stats"
    | "complete"
    | "error"
    | "log";
  stage?: string;
  status?: string;
  total?: number;
  queries?: string[];
  focuses?: QuestionFocus[];
  sub_focuses?: QuestionFocus[];
  completed?: number;
  failed?: number;
  requested?: number;
  question_id?: string;
  focus?: string | QuestionFocus;
  round?: number;
  question?: {
    question: string;
    correct_answer: string;
    explanation: string;
  };
  validation?: { decision?: string };
  extended?: boolean;
  rounds?: number;
  index?: number;
  error?: string;
  reason?: string;
  content?: string;
  all_agents?: unknown;
  stats?: { tokens?: number };
}
