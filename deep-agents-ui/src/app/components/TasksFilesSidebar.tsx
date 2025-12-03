"use client";

import React, {
  useMemo,
  useCallback,
  useState,
  useEffect,
  useRef,
} from "react";
import {
  FileText,
  FileSpreadsheet,
  FileImage,
  File,
  CheckCircle,
  Circle,
  Clock,
  ChevronDown,
  Download,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { TodoItem, FileItem } from "@/app/types/types";
import { useChatContext } from "@/providers/ChatProvider";
import { cn } from "@/lib/utils";
import { FileViewDialog } from "@/app/components/FileViewDialog";
import { BinaryFileDialog } from "@/app/components/BinaryFileDialog";
import type { BinaryFileData, FileValue } from "@/app/hooks/useChat";

// Helper to check if file data is binary
function isBinaryFile(fileData: unknown): fileData is BinaryFileData {
  const result = (
    typeof fileData === "object" &&
    fileData !== null &&
    "is_binary" in fileData &&
    (fileData as BinaryFileData).is_binary === true
  );
  // Debug logging
  if (typeof fileData === "object" && fileData !== null) {
    console.log("[isBinaryFile] Checking file:", {
      hasIsBinary: "is_binary" in fileData,
      isBinaryValue: (fileData as BinaryFileData).is_binary,
      hasContentBase64: "content_base64" in fileData,
      result,
    });
  }
  return result;
}

// Helper to get file icon based on extension
function getFileIcon(filePath: string, isBinary: boolean) {
  const ext = filePath.split(".").pop()?.toLowerCase() || "";
  
  if (isBinary) {
    switch (ext) {
      case "xlsx":
      case "xls":
        return <FileSpreadsheet size={24} className="mx-auto text-green-600" />;
      case "pptx":
      case "ppt":
        return <FileImage size={24} className="mx-auto text-orange-500" />;
      case "pdf":
        return <File size={24} className="mx-auto text-red-500" />;
      case "docx":
      case "doc":
        return <FileText size={24} className="mx-auto text-blue-500" />;
      default:
        return <File size={24} className="mx-auto text-muted-foreground" />;
    }
  }
  
  return <FileText size={24} className="mx-auto text-muted-foreground" />;
}

// Helper to download binary file
function downloadBinaryFile(filePath: string, fileData: BinaryFileData) {
  const fileName = filePath.split("/").pop() || "download";
  
  // Prefer download_url if available (from Supabase Storage)
  if (fileData.download_url) {
    window.open(fileData.download_url, "_blank");
    return;
  }
  
  // Fallback to base64 if available
  if (fileData.content_base64) {
    const byteCharacters = atob(fileData.content_base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: fileData.content_type });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

// Format file size
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Selected binary file state type
interface SelectedBinaryFile {
  path: string;
  data: BinaryFileData;
}

export function FilesPopover({
  files,
  setFiles,
  editDisabled,
}: {
  files: Record<string, FileValue>;
  setFiles: (files: Record<string, FileValue>) => Promise<void>;
  editDisabled: boolean;
}) {
  const [selectedFile, setSelectedFile] = useState<FileItem | null>(null);
  const [selectedBinaryFile, setSelectedBinaryFile] = useState<SelectedBinaryFile | null>(null);

  const handleSaveFile = useCallback(
    async (fileName: string, content: string) => {
      await setFiles({ ...files, [fileName]: content });
      setSelectedFile({ path: fileName, content: content });
    },
    [files, setFiles]
  );

  return (
    <>
      {Object.keys(files).length === 0 ? (
        <div className="flex h-full items-center justify-center p-4 text-center">
          <p className="text-xs text-muted-foreground">No files created yet</p>
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(256px,1fr))] gap-2">
          {Object.keys(files).map((file) => {
            const filePath = String(file);
            const rawContent = files[file];
            const isBinary = isBinaryFile(rawContent);
            
            let fileContent: string;
            if (isBinary) {
              // Binary file - show placeholder content
              fileContent = rawContent.content || `[Binary file - ${formatFileSize(rawContent.size)}]`;
            } else if (
              typeof rawContent === "object" &&
              rawContent !== null &&
              "content" in rawContent
            ) {
              const contentArray = (rawContent as { content: unknown }).content;
              if (Array.isArray(contentArray)) {
                fileContent = contentArray.join("\n");
              } else {
                fileContent = String(contentArray || "");
              }
            } else {
              fileContent = String(rawContent || "");
            }

            return (
              <div
                key={filePath}
                className="relative cursor-pointer space-y-1 truncate rounded-md border border-border px-2 py-3 shadow-sm transition-colors"
                style={{
                  backgroundColor: "var(--color-file-button)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--color-file-button-hover)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--color-file-button)";
                }}
              >
                {/* File icon and name - clickable */}
                <button
                  type="button"
                  onClick={() => {
                    if (isBinary) {
                      // For binary files, open preview dialog
                      setSelectedBinaryFile({ path: filePath, data: rawContent as BinaryFileData });
                    } else {
                      // For text files, open viewer
                      setSelectedFile({ path: filePath, content: fileContent });
                    }
                  }}
                  className="w-full"
                >
                  {getFileIcon(filePath, isBinary)}
                  <span className="mx-auto block w-full truncate break-words text-center text-sm leading-relaxed text-foreground">
                    {filePath.split("/").pop()}
                  </span>
                  {isBinary && (
                    <span className="mx-auto block text-xs text-muted-foreground">
                      {formatFileSize((rawContent as BinaryFileData).size)}
                    </span>
                  )}
                </button>
                
                {/* Download button for binary files */}
                {isBinary && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      downloadBinaryFile(filePath, rawContent as BinaryFileData);
                    }}
                    className="absolute right-1 top-1 rounded-full bg-primary/10 p-1.5 text-primary hover:bg-primary/20"
                    title="Download file"
                  >
                    <Download size={14} />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Text file viewer dialog */}
      {selectedFile && (
        <FileViewDialog
          file={selectedFile}
          onSaveFile={handleSaveFile}
          onClose={() => setSelectedFile(null)}
          editDisabled={editDisabled}
        />
      )}

      {/* Binary file preview dialog */}
      {selectedBinaryFile && (
        <BinaryFileDialog
          filePath={selectedBinaryFile.path}
          fileData={selectedBinaryFile.data}
          onClose={() => setSelectedBinaryFile(null)}
        />
      )}
    </>
  );
}

export const TasksFilesSidebar = React.memo<{
  todos: TodoItem[];
  files: Record<string, FileValue>;
  setFiles: (files: Record<string, FileValue>) => Promise<void>;
}>(({ todos, files, setFiles }) => {
  const { isLoading, interrupt } = useChatContext();
  const [tasksOpen, setTasksOpen] = useState(false);
  const [filesOpen, setFilesOpen] = useState(false);

  // Track previous counts to detect when content goes from empty to having items
  const prevTodosCount = useRef(todos.length);
  const prevFilesCount = useRef(Object.keys(files).length);

  // Auto-expand when todos go from empty to having content
  useEffect(() => {
    if (prevTodosCount.current === 0 && todos.length > 0) {
      setTasksOpen(true);
    }
    prevTodosCount.current = todos.length;
  }, [todos.length]);

  // Auto-expand when files go from empty to having content
  const filesCount = Object.keys(files).length;
  useEffect(() => {
    if (prevFilesCount.current === 0 && filesCount > 0) {
      setFilesOpen(true);
    }
    prevFilesCount.current = filesCount;
  }, [filesCount]);

  const getStatusIcon = useCallback((status: TodoItem["status"]) => {
    switch (status) {
      case "completed":
        return (
          <CheckCircle
            size={12}
            className="text-success/80"
          />
        );
      case "in_progress":
        return (
          <Clock
            size={12}
            className="text-warning/80"
          />
        );
      default:
        return (
          <Circle
            size={10}
            className="text-tertiary/70"
          />
        );
    }
  }, []);

  const groupedTodos = useMemo(() => {
    return {
      pending: todos.filter((t) => t.status === "pending"),
      in_progress: todos.filter((t) => t.status === "in_progress"),
      completed: todos.filter((t) => t.status === "completed"),
    };
  }, [todos]);

  const groupedLabels = {
    pending: "Pending",
    in_progress: "In Progress",
    completed: "Completed",
  };

  return (
    <div className="min-h-0 w-full flex-1">
      <div className="font-inter flex h-full w-full flex-col p-0">
        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden">
          <div className="flex items-center justify-between px-3 pb-1.5 pt-2">
            <span className="text-xs font-semibold tracking-wide text-zinc-600">
              AGENT TASKS
            </span>
            <button
              onClick={() => setTasksOpen((v) => !v)}
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-transform duration-200 hover:bg-muted",
                tasksOpen ? "rotate-180" : "rotate-0"
              )}
              aria-label="Toggle tasks panel"
            >
              <ChevronDown size={14} />
            </button>
          </div>
          {tasksOpen && (
            <div className="bg-muted-secondary rounded-xl px-3 pb-2">
              <ScrollArea className="h-full">
                {todos.length === 0 ? (
                  <div className="flex h-full items-center justify-center p-4 text-center">
                    <p className="text-xs text-muted-foreground">
                      No tasks created yet
                    </p>
                  </div>
                ) : (
                  <div className="ml-1 p-0.5">
                    {Object.entries(groupedTodos).map(([status, todos]) => (
                      <div className="mb-4">
                        <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-tertiary">
                          {groupedLabels[status as keyof typeof groupedLabels]}
                        </h3>
                        {todos.map((todo, index) => (
                          <div
                            key={`${status}_${todo.id}_${index}`}
                            className="mb-1.5 flex items-start gap-2 rounded-sm p-1 text-sm"
                          >
                            {getStatusIcon(todo.status)}
                            <span className="flex-1 break-words leading-relaxed text-inherit">
                              {todo.content}
                            </span>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </div>
          )}

          <div className="flex items-center justify-between px-3 pb-1.5 pt-2">
            <span className="text-xs font-semibold tracking-wide text-zinc-600">
              FILE SYSTEM
            </span>
            <button
              onClick={() => setFilesOpen((v) => !v)}
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-transform duration-200 hover:bg-muted",
                filesOpen ? "rotate-180" : "rotate-0"
              )}
              aria-label="Toggle files panel"
            >
              <ChevronDown size={14} />
            </button>
          </div>
          {filesOpen && (
            <FilesPopover
              files={files}
              setFiles={setFiles}
              editDisabled={isLoading === true || interrupt !== undefined}
            />
          )}
        </div>
      </div>
    </div>
  );
});

TasksFilesSidebar.displayName = "TasksFilesSidebar";
