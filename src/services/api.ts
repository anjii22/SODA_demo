const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export interface ArticulationError {
  expected: string;
  produced: string;
  details: {
    manner_error?: string;
    place_error?: string;
    voicing_error?: string;
  };
}

export interface SodaErrors {
  base_soda_error: string;
  primary_error_type: string;
}

export interface TonguePositionEntry {
  expected_phoneme: string;
  produced_phoneme: string;
  expected_tongue_position: string;
  produced_tongue_position: string;
}

export interface AnalysisResult {
  is_correct: boolean;
  wrong_word: boolean;
  severity: number;
  severity_phoneme: number;
  severity_text: number;
  acoustic_severity: number;
  articulation_errors: ArticulationError[];
  soda_errors: SodaErrors;
  tongue_position_analysis: TonguePositionEntry[];
  predicted_word: string;
  whisper_word: string;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const analyzeSpeech = async (
  file: File,
  expectedText: string
): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append("audio", file);
  formData.append("expected_text", expectedText);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Server error: ${response.status}`;
    try {
      const body = await response.json();
      if (body?.error) message = body.error;
      else if (body?.message) message = body.message;
    } catch {
      // use default message
    }
    throw new ApiError(response.status, message);
  }

  return response.json() as Promise<AnalysisResult>;
};
