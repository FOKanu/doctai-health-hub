import { BaseApiService, ApiResponse } from './baseApiService';

export interface OpenAIConfig {
  apiKey: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface OpenAIRequest {
  prompt: string;
  systemMessage?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface OpenAIResponse {
  text: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  model: string;
}

export interface HealthInsightRequest {
  symptoms: string[];
  age?: number;
  gender?: string;
  medicalHistory?: string[];
  currentMedications?: string[];
}

export interface HealthInsightResponse {
  possibleConditions: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
  urgency: 'routine' | 'soon' | 'urgent' | 'emergency';
  explanation: string;
  disclaimer: string;
}

interface OpenAIApiResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export class OpenAIService extends BaseApiService {
  private apiKey: string;
  private model: string;
  private maxTokens: number;
  private temperature: number;

  constructor(config: OpenAIConfig) {
    super({
      baseURL: 'https://api.openai.com/v1',
      headers: {
        'Authorization': `Bearer ${config.apiKey}`
      }
    });

    this.apiKey = config.apiKey;
    this.model = config.model || 'gpt-4';
    this.maxTokens = config.maxTokens || 1000;
    this.temperature = config.temperature || 0.7;
  }

  async generateText(request: OpenAIRequest): Promise<ApiResponse<OpenAIResponse>> {
    const messages = [];

    if (request.systemMessage) {
      messages.push({
        role: 'system',
        content: request.systemMessage
      });
    }

    messages.push({
      role: 'user',
      content: request.prompt
    });

    const response = await this.post('/chat/completions', {
      model: this.model,
      messages,
      max_tokens: request.maxTokens || this.maxTokens,
      temperature: request.temperature || this.temperature
    });

    if (response.success && response.data) {
      const apiData = response.data as OpenAIApiResponse;
      const choice = apiData.choices?.[0];
      if (choice) {
        return {
          ...response,
          data: {
            text: choice.message.content,
            usage: {
              promptTokens: apiData.usage.prompt_tokens,
              completionTokens: apiData.usage.completion_tokens,
              totalTokens: apiData.usage.total_tokens
            },
            model: this.model
          }
        };
      }
    }

    return {
      data: null,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      success: false,
      error: 'Failed to generate text'
    };
  }

  async analyzeHealthInsights(request: HealthInsightRequest): Promise<ApiResponse<HealthInsightResponse>> {
    const systemMessage = `You are a medical AI assistant. Provide health insights based on symptoms and patient information.
    Always include appropriate disclaimers and recommend consulting healthcare professionals for serious concerns.
    Format your response as JSON with the following structure:
    {
      "possibleConditions": ["condition1", "condition2"],
      "severity": "low|medium|high|critical",
      "recommendations": ["recommendation1", "recommendation2"],
      "urgency": "routine|soon|urgent|emergency",
      "explanation": "detailed explanation",
      "disclaimer": "medical disclaimer"
    }`;

    const prompt = `Analyze the following health information:
    Symptoms: ${request.symptoms.join(', ')}
    Age: ${request.age || 'Not specified'}
    Gender: ${request.gender || 'Not specified'}
    Medical History: ${request.medicalHistory?.join(', ') || 'None'}
    Current Medications: ${request.currentMedications?.join(', ') || 'None'}

    Provide a comprehensive health analysis.`;

    const response = await this.generateText({
      prompt,
      systemMessage,
      maxTokens: 1500,
      temperature: 0.3
    });

    if (response.success && response.data) {
      try {
        const parsedData = JSON.parse(response.data.text) as HealthInsightResponse;
        return {
          data: parsedData,
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
          success: true
        };
      } catch (error) {
        return {
          data: null,
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
          success: false,
          error: 'Failed to parse AI response'
        };
      }
    }

    return {
      data: null,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      success: false,
      error: response.error || 'Failed to analyze health insights'
    };
  }

  async explainMedicalTerm(term: string): Promise<ApiResponse<{ explanation: string; simplified: string }>> {
    const systemMessage = `You are a medical educator. Explain medical terms in simple, understandable language.
    Provide both a detailed explanation and a simplified version for patients.`;

    const prompt = `Explain the medical term: "${term}"

    Provide:
    1. A detailed medical explanation
    2. A simplified explanation for patients`;

    const response = await this.generateText({
      prompt,
      systemMessage,
      maxTokens: 800,
      temperature: 0.5
    });

    if (response.success && response.data) {
      // Parse the response to extract detailed and simplified explanations
      const text = response.data.text;
      const lines = text.split('\n');
      let detailed = '';
      let simplified = '';

      let currentSection = '';
      for (const line of lines) {
        if (line.toLowerCase().includes('detailed') || line.toLowerCase().includes('medical')) {
          currentSection = 'detailed';
        } else if (line.toLowerCase().includes('simplified') || line.toLowerCase().includes('patient')) {
          currentSection = 'simplified';
        } else if (line.trim()) {
          if (currentSection === 'detailed') {
            detailed += line + '\n';
          } else if (currentSection === 'simplified') {
            simplified += line + '\n';
          }
        }
      }

      return {
        ...response,
        data: {
          explanation: detailed.trim(),
          simplified: simplified.trim()
        }
      };
    }

    return {
      data: null,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      success: false,
      error: response.error || 'Failed to explain medical term'
    };
  }

  async generateHealthTips(category: string, count: number = 5): Promise<ApiResponse<string[]>> {
    const systemMessage = `You are a health and wellness expert. Generate practical health tips for the given category.
    Make tips actionable, evidence-based, and easy to follow.`;

    const prompt = `Generate ${count} health tips for: ${category}

    Format as a simple list, one tip per line.`;

    const response = await this.generateText({
      prompt,
      systemMessage,
      maxTokens: 600,
      temperature: 0.7
    });

    if (response.success && response.data) {
      const tips = response.data.text
        .split('\n')
        .map(tip => tip.trim())
        .filter(tip => tip && !tip.match(/^\d+\./))
        .slice(0, count);

      return {
        ...response,
        data: tips
      };
    }

    return {
      data: null,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      success: false,
      error: response.error || 'Failed to generate health tips'
    };
  }

  async summarizeMedicalReport(report: string): Promise<ApiResponse<{ summary: string; keyPoints: string[] }>> {
    const systemMessage = `You are a medical professional. Summarize medical reports in clear, concise language.
    Extract key points and provide a brief summary.`;

    const prompt = `Summarize this medical report:

    ${report}

    Provide:
    1. A brief summary
    2. Key points as a list`;

    const response = await this.generateText({
      prompt,
      systemMessage,
      maxTokens: 1000,
      temperature: 0.3
    });

    if (response.success && response.data) {
      const text = response.data.text;
      const lines = text.split('\n');
      let summary = '';
      const keyPoints: string[] = [];

      let currentSection = '';
      for (const line of lines) {
        if (line.toLowerCase().includes('summary')) {
          currentSection = 'summary';
        } else if (line.toLowerCase().includes('key points') || line.toLowerCase().includes('points')) {
          currentSection = 'points';
        } else if (line.trim()) {
          if (currentSection === 'summary') {
            summary += line + '\n';
          } else if (currentSection === 'points') {
            if (line.match(/^[-•*]\s/) || line.match(/^\d+\./)) {
              keyPoints.push(line.replace(/^[-•*]\s*/, '').replace(/^\d+\.\s*/, ''));
            }
          }
        }
      }

      return {
        ...response,
        data: {
          summary: summary.trim(),
          keyPoints
        }
      };
    }

    return {
      data: null,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      success: false,
      error: response.error || 'Failed to summarize medical report'
    };
  }
}
