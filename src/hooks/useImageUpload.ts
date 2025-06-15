import { useState } from 'react';
import { storageService, ImageMetadata } from '../services/storageService';

interface UseImageUploadOptions {
  onSuccess?: (metadata: ImageMetadata) => void;
  onError?: (error: Error) => void;
}

export const useImageUpload = (options: UseImageUploadOptions = {}) => {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<Error | null>(null);

  const uploadImage = async (
    file: File | Blob,
    type: ImageMetadata['type'],
    metadata: Partial<ImageMetadata['metadata']> = {}
  ) => {
    try {
      setIsUploading(true);
      setError(null);
      setProgress(0);

      // Add device info to metadata
      const enhancedMetadata = {
        ...metadata,
        device_info: {
          model: navigator.userAgent,
          os: navigator.platform,
          browser: navigator.userAgent
        }
      };

      const result = await storageService.uploadImage(file, type, enhancedMetadata);

      setProgress(100);
      options.onSuccess?.(result);
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Upload failed');
      setError(error);
      options.onError?.(error);
      throw error;
    } finally {
      setIsUploading(false);
    }
  };

  return {
    uploadImage,
    isUploading,
    progress,
    error
  };
};
