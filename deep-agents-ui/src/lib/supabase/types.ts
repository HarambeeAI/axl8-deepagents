/**
 * Database types for Supabase tables.
 * These types match the schema created in the Supabase migrations.
 */

export interface User {
  id: string;
  email: string | null;
  display_name: string | null;
  avatar_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface Thread {
  id: string;
  user_id: string | null;
  title: string | null;
  assistant_id: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  thread_id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string | null;
  tool_calls: unknown[] | null;
  tool_call_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface AgentFile {
  id: string;
  thread_id: string;
  path: string;
  content: string | null;
  content_type: string;
  size_bytes: number | null;
  created_at: string;
  updated_at: string;
}

export interface Database {
  public: {
    Tables: {
      users: {
        Row: User;
        Insert: Omit<User, "id" | "created_at" | "updated_at">;
        Update: Partial<Omit<User, "id" | "created_at" | "updated_at">>;
      };
      threads: {
        Row: Thread;
        Insert: Omit<Thread, "id" | "created_at" | "updated_at">;
        Update: Partial<Omit<Thread, "id" | "created_at" | "updated_at">>;
      };
      messages: {
        Row: Message;
        Insert: Omit<Message, "id" | "created_at">;
        Update: Partial<Omit<Message, "id" | "created_at">>;
      };
      agent_files: {
        Row: AgentFile;
        Insert: Omit<AgentFile, "id" | "created_at" | "updated_at">;
        Update: Partial<Omit<AgentFile, "id" | "created_at" | "updated_at">>;
      };
    };
  };
}
