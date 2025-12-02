"use client";

import React, { useCallback, useEffect, useMemo } from "react";
import {
  FileSpreadsheet,
  FileImage,
  File,
  FileText,
  Download,
  ExternalLink,
} from "lucide-react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

// Binary file type definition
interface BinaryFileData {
  content: string;
  is_binary: boolean;
  content_base64: string;
  content_type: string;
  size: number;
}

interface BinaryFileDialogProps {
  filePath: string;
  fileData: BinaryFileData;
  onClose: () => void;
}

// Helper to get file icon based on extension
function getFileIcon(filePath: string) {
  const ext = filePath.split(".").pop()?.toLowerCase() || "";
  
  switch (ext) {
    case "xlsx":
    case "xls":
      return <FileSpreadsheet size={48} className="text-green-600" />;
    case "pptx":
    case "ppt":
      return <FileImage size={48} className="text-orange-500" />;
    case "pdf":
      return <File size={48} className="text-red-500" />;
    case "docx":
    case "doc":
      return <FileText size={48} className="text-blue-500" />;
    default:
      return <File size={48} className="text-muted-foreground" />;
  }
}

// Format file size
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Get document type name
function getDocTypeName(filePath: string): string {
  const ext = filePath.split(".").pop()?.toLowerCase() || "";
  switch (ext) {
    case "xlsx":
    case "xls":
      return "Excel Spreadsheet";
    case "pptx":
    case "ppt":
      return "PowerPoint Presentation";
    case "pdf":
      return "PDF Document";
    case "docx":
    case "doc":
      return "Word Document";
    default:
      return "Binary File";
  }
}

export const BinaryFileDialog = React.memo<BinaryFileDialogProps>(
  ({ filePath, fileData, onClose }) => {
    const fileName = filePath.split("/").pop() || "document";
    const ext = filePath.split(".").pop()?.toLowerCase() || "";

    // Download the file
    const handleDownload = useCallback(() => {
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
    }, [fileData, fileName]);

    // Create a blob URL and open in new tab (for PDF)
    const handleOpenInNewTab = useCallback(() => {
      const byteCharacters = atob(fileData.content_base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: fileData.content_type });
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
    }, [fileData]);

    // For PDF files, we can embed them directly
    const isPdf = ext === "pdf";
    
    // Create blob URL for PDF preview
    const pdfBlobUrl = useMemo(() => {
      if (!isPdf) return null;
      try {
        const byteCharacters = atob(fileData.content_base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: "application/pdf" });
        return URL.createObjectURL(blob);
      } catch {
        return null;
      }
    }, [isPdf, fileData.content_base64]);

    // Cleanup blob URL on unmount
    useEffect(() => {
      return () => {
        if (pdfBlobUrl) {
          URL.revokeObjectURL(pdfBlobUrl);
        }
      };
    }, [pdfBlobUrl]);

    return (
      <Dialog open={true} onOpenChange={onClose}>
        <DialogContent className="flex h-[80vh] max-h-[80vh] min-w-[60vw] flex-col p-6">
          <DialogTitle className="sr-only">{fileName}</DialogTitle>
          
          {/* Header */}
          <div className="mb-4 flex items-center justify-between border-b border-border pb-4">
            <div className="flex items-center gap-3">
              {getFileIcon(filePath)}
              <div>
                <h2 className="text-lg font-semibold text-primary">{fileName}</h2>
                <p className="text-sm text-muted-foreground">
                  {getDocTypeName(filePath)} • {formatFileSize(fileData.size)}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                onClick={handleDownload}
                variant="default"
                size="sm"
                className="h-9"
              >
                <Download size={16} className="mr-2" />
                Download
              </Button>
              {isPdf && (
                <Button
                  onClick={handleOpenInNewTab}
                  variant="outline"
                  size="sm"
                  className="h-9"
                >
                  <ExternalLink size={16} className="mr-2" />
                  Open in New Tab
                </Button>
              )}
            </div>
          </div>

          {/* Content */}
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-hidden rounded-lg bg-muted/30">
            {isPdf && pdfBlobUrl ? (
              // PDF Preview using iframe
              <iframe
                src={pdfBlobUrl}
                className="h-full w-full rounded-lg"
                title={fileName}
              />
            ) : (
              // Non-PDF files - show download prompt
              <div className="flex flex-col items-center gap-6 p-12 text-center">
                {getFileIcon(filePath)}
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-2">
                    {getDocTypeName(filePath)}
                  </h3>
                  <p className="text-muted-foreground mb-1">
                    {fileName}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(fileData.size)}
                  </p>
                </div>
                <div className="flex flex-col gap-3">
                  <Button
                    onClick={handleDownload}
                    size="lg"
                    className="gap-2"
                  >
                    <Download size={20} />
                    Download File
                  </Button>
                  <p className="text-xs text-muted-foreground max-w-sm">
                    This document was generated using Claude Skills. 
                    Download it to open in {ext === "xlsx" ? "Excel" : ext === "pptx" ? "PowerPoint" : ext === "docx" ? "Word" : "your preferred application"}.
                  </p>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    );
  }
);

BinaryFileDialog.displayName = "BinaryFileDialog";
