import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

export interface ApiConfig {
  baseURL: string;
  timeout?: number;
  headers?: Record<string, string>;
  retries?: number;
  retryDelay?: number;
}

export interface ApiResponse<T = unknown> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, unknown>;
  success: boolean;
  error?: string;
}

export class BaseApiService {
  protected client: AxiosInstance;
  protected config: ApiConfig;

  constructor(config: ApiConfig) {
    this.config = {
      timeout: 30000,
      retries: 3,
      retryDelay: 1000,
      ...config
    };

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...this.config.headers
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor for authentication
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized - refresh token or redirect to login
          await this.handleUnauthorized();
        }
        return Promise.reject(error);
      }
    );
  }

  protected async request<T>(
    config: AxiosRequestConfig,
    retryCount = 0
  ): Promise<ApiResponse<T>> {
    try {
      const response: AxiosResponse<T> = await this.client.request(config);
      return {
        data: response.data,
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
        success: true
      };
    } catch (error: Error | unknown) {
      if (retryCount < this.config.retries! && this.shouldRetry(error)) {
        await this.delay(this.config.retryDelay!);
        return this.request(config, retryCount + 1);
      }

      return {
        data: null as T,
        status: error.response?.status || 0,
        statusText: error.response?.statusText || 'Network Error',
        headers: error.response?.headers || {},
        success: false,
        error: error.message
      };
    }
  }

  protected shouldRetry(error: Error | unknown): boolean {
    const status = error.response?.status;
    return status >= 500 || status === 429 || !status; // Server errors, rate limit, network errors
  }

  protected delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  protected getAuthToken(): string | null {
    // Get token from localStorage, sessionStorage, or other storage
    return localStorage.getItem('auth_token');
  }

  protected async handleUnauthorized(): Promise<void> {
    // Clear token and redirect to login
    localStorage.removeItem('auth_token');
    window.location.href = '/login';
  }

  // Utility methods
  protected async get<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'GET', url });
  }

  protected async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'POST', url, data });
  }

  protected async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'PUT', url, data });
  }

  protected async delete<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'DELETE', url });
  }

  protected async patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'PATCH', url, data });
  }
}
