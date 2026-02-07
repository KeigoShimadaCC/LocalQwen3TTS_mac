import { useMemo, useState } from "react";
import { synthesize } from "./api";
import type { TTSRequestPayload, TTSResponse } from "./api";

const models = [
  { label: "Qwen3-TTS 0.6B", value: "qwen3-tts-0.6b" },
  { label: "Qwen3-TTS 1.7B", value: "qwen3-tts-1.7b" },
];

const languages = [
  { label: "Auto", value: "auto" },
  { label: "English", value: "en" },
  { label: "Japanese", value: "ja" },
  { label: "Chinese", value: "zh" },
  { label: "French", value: "fr" },
  { label: "Spanish", value: "es" },
  { label: "German", value: "de" },
  { label: "Korean", value: "ko" },
];

const voiceOptions = ["custom_female", "custom_male", "storyteller"];

const examples = [
  {
    text: "Welcome to the Qwen3 TTS demo!",
    voice: "custom_female",
    language: "en",
    model: "qwen3-tts-0.6b",
    tone: "cheerful and warm",
  },
  {
    text: "今日も頑張っていきましょう。",
    voice: "storyteller",
    language: "ja",
    model: "qwen3-tts-1.7b",
    tone: "calm narrator",
  },
  {
    text: "La technologie vocale progresse rapidement.",
    voice: "custom_male",
    language: "fr",
    model: "qwen3-tts-0.6b",
    tone: "informative",
  },
];

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

function App() {
  const [text, setText] = useState("Welcome to Qwen3-TTS running on macOS!");
  const [model, setModel] = useState(models[0].value);
  const [voice, setVoice] = useState(voiceOptions[0]);
  const [language, setLanguage] = useState(languages[0].value);
  const [tone, setTone] = useState("cheerful and warm");
  const [format, setFormat] = useState("wav");
  const [speed, setSpeed] = useState(1.0);
  const [requestId, setRequestId] = useState<string | undefined>();
  const [audioSrc, setAudioSrc] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();

  const voiceList = useMemo(() => voiceOptions, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(undefined);
    setAudioSrc(undefined);
    try {
      const payload: TTSRequestPayload = {
        text,
        model,
        voice,
        language,
        tone,
        format: format as "wav" | "mp3",
        speed,
      };
      const response = await synthesize(API_BASE, payload);
      setRequestId(response.request_id);
      if (response.audio_base64) {
        const binary = atob(response.audio_base64);
        const array = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          array[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([array], { type: `audio/${response.audio_format}` });
        setAudioSrc(URL.createObjectURL(blob));
      } else if (response.audio_url) {
        setAudioSrc(`${API_BASE}${response.audio_url}`);
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? err.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (example: (typeof examples)[0]) => {
    setText(example.text);
    setVoice(example.voice);
    setLanguage(example.language);
    setModel(example.model);
    setTone(example.tone);
  };

  return (
    <div className="page">
      <header>
        <h1>Qwen3-TTS Control Room</h1>
        <p>Run CustomVoice models locally on Apple Silicon.</p>
      </header>
      <div className="content">
        <form className="controls" onSubmit={handleSubmit}>
          <label>
            Text to Synthesize
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={6} required />
          </label>
          <div className="grid">
            <label>
              Model
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {models.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Voice
              <select value={voice} onChange={(e) => setVoice(e.target.value)}>
                {voiceList.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Language
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                {languages.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Tone / Instruction
              <input value={tone} onChange={(e) => setTone(e.target.value)} placeholder="cheerful" />
            </label>
            <label>
              Output Format
              <select value={format} onChange={(e) => setFormat(e.target.value)}>
                <option value="wav">WAV (default)</option>
                <option value="mp3">MP3</option>
              </select>
            </label>
            <label>
              Speed
              <input
                type="number"
                min={0.5}
                max={2}
                step={0.1}
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
              />
            </label>
          </div>
          <button type="submit" disabled={loading}>
            {loading ? "Generating..." : "Generate"}
          </button>
          {error && <p className="error">{error}</p>}
          {requestId && <p className="request">request_id: {requestId}</p>}
        </form>
        <div className="preview">
          <div className="player">
            <h2>Playback</h2>
            {audioSrc ? <audio controls src={audioSrc} /> : <p>No audio yet.</p>}
          </div>
          <div className="examples">
            <h3>Examples</h3>
            <table>
              <thead>
                <tr>
                  <th>Text</th>
                  <th>Voice</th>
                  <th>Language</th>
                  <th>Model</th>
                  <th>Tone</th>
                </tr>
              </thead>
              <tbody>
                {examples.map((ex, idx) => (
                  <tr key={idx} onClick={() => handleExample(ex)}>
                    <td>{ex.text}</td>
                    <td>{ex.voice}</td>
                    <td>{ex.language}</td>
                    <td>{ex.model}</td>
                    <td>{ex.tone}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
