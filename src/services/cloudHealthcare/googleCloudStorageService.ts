import { CloudAnalysisResult, ImageType } from './types';

export interface GoogleCloudStorageConfig {
  projectId: string;
  bucketName: string;
  location?: string;
  apiKey?: string;
}

export interface StorageMetadata {
  contentType: string;
  metadata?: Record<string, string>;
  cacheControl?: string;
}

export class GoogleCloudStorageService {
  private projectId: string;
  private bucketName: string;
  private location: string;
  private apiKey?: string;
  private baseUrl: string;

  constructor(config: GoogleCloudStorageConfig) {
    this.projectId = config.projectId;
    this.bucketName = config.bucketName;
    this.location = config.location || 'us-central1';
    this.apiKey = config.apiKey;
    this.baseUrl = `https://storage.googleapis.com/storage/v1/b/${this.bucketName}/o`;
  }

  /**
   * Upload a file to Google Cloud Storage
   */
  async uploadFile(
    file: File | Blob,
    path: string,
    metadata: StorageMetadata = { contentType: 'application/octet-stream' }
  ): Promise<{ url: string; path: string }> {
    try {
      // Convert file to base64 for upload
      const base64Data = await this.fileToBase64(file);

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const requestBody = {
        name: path,
        media: {
          mimeType: metadata.contentType,
          body: base64Data
        },
        metadata: metadata.metadata || {}
      };

      const response = await fetch(`${this.baseUrl}?uploadType=multipart`, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      const publicUrl = `https://storage.googleapis.com/${this.bucketName}/${path}`;

      return {
        url: publicUrl,
        path: path
      };
    } catch (error) {
      console.error('Google Cloud Storage upload error:', error);
      throw new Error(`Upload to Google Cloud Storage failed: ${error.message}`);
    }
  }

  /**
   * Upload a medical image with proper organization
   */
  async uploadMedicalImage(
    file: File,
    imageType: ImageType,
    userId: string,
    analysisResult?: CloudAnalysisResult
  ): Promise<{ url: string; path: string; metadata: unknown }> {
    const timestamp = new Date().toISOString();
    const fileExt = file.name.split('.').pop() || 'jpg';
    const fileName = `${timestamp}-${Math.random().toString(36).substr(2, 9)}.${fileExt}`;

    // Organize by user and image type
    const path = `users/${userId}/${imageType}/${fileName}`;

    const metadata: StorageMetadata = {
      contentType: file.type,
      metadata: {
        userId,
        imageType,
        originalName: file.name,
        uploadTimestamp: timestamp,
        analysisResult: analysisResult ? JSON.stringify(analysisResult) : '',
        riskLevel: analysisResult?.riskLevel || 'unknown'
      }
    };

    const result = await this.uploadFile(file, path, metadata);

    return {
      ...result,
      metadata: {
        ...metadata.metadata,
        size: file.size,
        type: imageType
      }
    };
  }

  /**
   * Download a file from Google Cloud Storage
   */
  async downloadFile(path: string): Promise<Blob> {
    try {
      const headers: Record<string, string> = {};

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(`${this.baseUrl}/${encodeURIComponent(path)}?alt=media`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`Download failed: ${response.status} ${response.statusText}`);
      }

      return await response.blob();
    } catch (error) {
      console.error('Google Cloud Storage download error:', error);
      throw new Error(`Download from Google Cloud Storage failed: ${error.message}`);
    }
  }

  /**
   * Delete a file from Google Cloud Storage
   */
  async deleteFile(path: string): Promise<void> {
    try {
      const headers: Record<string, string> = {};

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(`${this.baseUrl}/${encodeURIComponent(path)}`, {
        method: 'DELETE',
        headers
      });

      if (!response.ok && response.status !== 404) {
        throw new Error(`Delete failed: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.error('Google Cloud Storage delete error:', error);
      throw new Error(`Delete from Google Cloud Storage failed: ${error.message}`);
    }
  }

  /**
   * List files in a directory
   */
  async listFiles(prefix?: string): Promise<Array<{ name: string; size: number; updated: string }>> {
    try {
      const headers: Record<string, string> = {};

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const url = new URL(this.baseUrl);
      if (prefix) {
        url.searchParams.set('prefix', prefix);
      }

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`List failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      return result.items?.map((item: unknown) => ({
        name: item.name,
        size: parseInt(item.size) || 0,
        updated: item.updated
      })) || [];
    } catch (error) {
      console.error('Google Cloud Storage list error:', error);
      throw new Error(`List from Google Cloud Storage failed: ${error.message}`);
    }
  }

  /**
   * Get user's medical images
   */
  async getUserMedicalImages(userId: string): Promise<Array<{ name: string; type: ImageType; url: string; metadata: unknown }>> {
    const prefix = `users/${userId}/`;
    const files = await this.listFiles(prefix);

    return files.map(file => {
      const pathParts = file.name.split('/');
      const imageType = pathParts[2] as ImageType; // users/userId/imageType/filename

      return {
        name: file.name,
        type: imageType,
        url: `https://storage.googleapis.com/${this.bucketName}/${file.name}`,
        metadata: {
          size: file.size,
          updated: file.updated
        }
      };
    });
  }

  /**
   * Create a signed URL for secure access
   */
  async createSignedUrl(path: string, expirationMinutes: number = 60): Promise<string> {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const expirationTime = new Date();
      expirationTime.setMinutes(expirationTime.getMinutes() + expirationMinutes);

      const requestBody = {
        name: path,
        expiration: expirationTime.toISOString()
      };

      const response = await fetch(`${this.baseUrl}/${encodeURIComponent(path)}/signedUrl`, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`Signed URL creation failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      return result.signedUrl;
    } catch (error) {
      console.error('Google Cloud Storage signed URL error:', error);
      throw new Error(`Signed URL creation failed: ${error.message}`);
    }
  }

  private async fileToBase64(file: File | Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64 = reader.result as string;
        // Remove data URL prefix
        resolve(base64.split(',')[1]);
      };
      reader.onerror = error => reject(error);
    });
  }

  /**
   * Get bucket information
   */
  getBucketInfo() {
    return {
      projectId: this.projectId,
      bucketName: this.bucketName,
      location: this.location,
      baseUrl: this.baseUrl
    };
  }
}
