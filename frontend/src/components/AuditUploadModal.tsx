import React, { useState } from 'react';
import { Upload, X, CheckCircle, AlertCircle, FileVideo, Loader2 } from 'lucide-react';

interface AuditUploadModalProps {
  onClose: () => void;
  apiBase: string;
}

export function AuditUploadModal({ onClose, apiBase }: AuditUploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
      setError('');
    } else {
      setError('Please upload a valid video file (MP4, AVI, etc.)');
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiBase}/api/audit/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error('Failed to upload video');
      }

      setSuccess('Video successfully uploaded. Retrospective audit started.');
      setTimeout(onClose, 3000);
    } catch (err: any) {
      setError(err.message || 'An error occurred during upload.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="audit-modal-backdrop" onClick={onClose}>
      <div className="audit-modal-content" onClick={e => e.stopPropagation()}>
        <div className="audit-modal-header">
          <h2>
            <Upload size={18} />
            Batch Video Upload
          </h2>
          <button className="btn-icon" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="audit-modal-body">
          {!success ? (
            <>
              <p>
                Upload raw surveillance footage or drone recordings for offline retrospective threat auditing.
                The video will be processed by the full AI pipeline at maximum speed.
              </p>

              <div
                className={`audit-dropzone ${file ? 'has-file' : ''}`}
                onDragOver={e => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => document.getElementById('audit-file-upload')?.click()}
              >
                <input
                  type="file"
                  id="audit-file-upload"
                  style={{ display: 'none' }}
                  accept="video/mp4,video/x-m4v,video/*"
                  onChange={e => {
                    const f = e.target.files?.[0];
                    if (f) { setFile(f); setError(''); }
                  }}
                />

                {file ? (
                  <>
                    <FileVideo size={48} />
                    <span className="audit-dropzone-title">{file.name}</span>
                    <span className="audit-dropzone-sub">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                  </>
                ) : (
                  <>
                    <Upload size={48} />
                    <span className="audit-dropzone-title">Click or drag video file here</span>
                    <span className="audit-dropzone-sub">Supports MP4, AVI, MKV (Max 500MB)</span>
                  </>
                )}
              </div>

              {error && (
                <div className="audit-error">
                  <AlertCircle size={16} />
                  <span>{error}</span>
                </div>
              )}

              <div className="audit-actions">
                <button className="btn btn-secondary" onClick={onClose} disabled={uploading}>
                  Cancel
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleUpload}
                  disabled={!file || uploading}
                >
                  {uploading ? (
                    <><Loader2 size={14} className="spin" /> Processing...</>
                  ) : 'Start Audit'}
                </button>
              </div>
            </>
          ) : (
            <div className="audit-success">
              <CheckCircle size={56} />
              <h3>Audit Initiated</h3>
              <p>
                The video is now being processed in the background.
                <br />Threats will appear in the Intelligence Panel marked as <strong>IMPORTED</strong>.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
