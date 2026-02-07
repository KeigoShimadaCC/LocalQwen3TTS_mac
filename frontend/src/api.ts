import axios from "axios";

export interface TTSRequestPayload {
  text: string;
  model: "qwen3-tts-0.6b" | "qwen3-tts-1.7b";
  voice?: string;
  language: string;
  tone?: string;
  format?: "wav" | "mp3";
  sample_rate?: number;
  seed?: number;
  speed?: number;
}

export interface TTSResponse {
  request_id: string;
  audio_format: "wav" | "mp3";
  sample_rate: number;
  duration_sec: number;
  audio_base64?: string;
  audio_url?: string;
}

export async function synthesize(baseUrl: string, payload: TTSRequestPayload): Promise<TTSResponse> {
  const response = await axios.post(`${baseUrl}/v1/tts`, payload);
  return response.data;
}
