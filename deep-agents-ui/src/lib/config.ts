export interface StandaloneConfig {
  deploymentUrl: string;
  assistantId: string;
  langsmithApiKey?: string;
}

const CONFIG_KEY = "deep-agent-config";

// Default configuration from environment variables
const DEFAULT_CONFIG: StandaloneConfig = {
  deploymentUrl:
    process.env.NEXT_PUBLIC_DEPLOYMENT_URL ||
    "https://worryless-deepagents.onrender.com",
  assistantId: process.env.NEXT_PUBLIC_ASSISTANT_ID || "agent",
  langsmithApiKey: process.env.NEXT_PUBLIC_LANGSMITH_API_KEY || undefined,
};

export function getConfig(): StandaloneConfig {
  if (typeof window === "undefined") return DEFAULT_CONFIG;

  const stored = localStorage.getItem(CONFIG_KEY);
  if (!stored) return DEFAULT_CONFIG;

  try {
    const parsed = JSON.parse(stored);
    // Merge with defaults to ensure all fields are present
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export function saveConfig(config: StandaloneConfig): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
}

export function getDefaultConfig(): StandaloneConfig {
  return DEFAULT_CONFIG;
}
