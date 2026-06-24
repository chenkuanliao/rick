import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Brain,
  CheckCircle2,
  Copy,
  LoaderCircle,
  MessageCircle,
  MessagesSquare,
  Mic,
  Pencil,
  Pin,
  PinOff,
  Play,
  Plus,
  Radio,
  Send,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Upload,
  Volume2,
  Wifi,
  WifiOff,
} from "lucide-react";
import "./styles.css";

type Phase = "idle" | "listening" | "transcribing" | "thinking" | "synthesizing" | "speaking" | "error";
type MobilePanel = "chat" | "chats" | "settings";
type ResponseActivity = { thinking: boolean; synthesizing: boolean; speaking: boolean };

type ProviderStatus = {
  name: string;
  model: string | null;
  configured: boolean;
  missing?: string[];
  disabled?: boolean;
  reason?: string;
  available_models?: string[];
};

type ConfigResponse = {
  default_provider: string;
  system_prompt: string;
  tts_output_prompt: string;
  providers: {
    default: string;
    providers: Record<string, ProviderStatus>;
  };
  tts: { voice_prompt_saved: boolean; loaded: boolean; available: boolean; error?: string | null };
};

type TranscriptItem = {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  time: string;
  provider?: string;
  model?: string | null;
  error?: boolean;
};

type ChatSummary = {
  id: string;
  title: string;
  pinned: boolean;
  created_at: string;
  updated_at: string;
  message_count: number;
};

type ChatDetail = ChatSummary & {
  messages: TranscriptItem[];
};

type BrowserMicStatus = {
  origin: string;
  protocol: string;
  secureContext: boolean;
  hasMediaDevices: boolean;
  hasGetUserMedia: boolean;
};

const phaseLabel: Record<Phase, string> = {
  idle: "Ready",
  listening: "Listening",
  transcribing: "Transcribing",
  thinking: "Thinking",
  synthesizing: "Synthesizing",
  speaking: "Speaking",
  error: "Needs attention"
};

const phaseDetail: Record<Phase, { label: string; detail: string; icon: React.ReactNode }> = {
  idle: {
    label: "Ready",
    detail: "Tap start, speak naturally, then tap stop.",
    icon: <CheckCircle2 size={20} />,
  },
  listening: {
    label: "Recording",
    detail: "Tap stop when you are done.",
    icon: <Mic size={20} />,
  },
  transcribing: {
    label: "Transcribing",
    detail: "Turning your voice into text.",
    icon: <Radio size={20} />,
  },
  thinking: {
    label: "Thinking",
    detail: "The model is composing a reply.",
    icon: <Brain size={20} />,
  },
  synthesizing: {
    label: "Generating voice",
    detail: "Preparing spoken audio.",
    icon: <Sparkles size={20} />,
  },
  speaking: {
    label: "Speaking",
    detail: "Playing the assistant response.",
    icon: <Volume2 size={20} />,
  },
  error: {
    label: "Needs attention",
    detail: "Check the latest system message.",
    icon: <LoaderCircle size={20} />,
  },
};

const SILENT_AUDIO_SRC =
  "data:audio/wav;base64,UklGRgQCAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YeABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";

function makeId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function getBrowserMicStatus(): BrowserMicStatus {
  return {
    origin: window.location.origin,
    protocol: window.location.protocol,
    secureContext: window.isSecureContext,
    hasMediaDevices: Boolean(navigator.mediaDevices),
    hasGetUserMedia: Boolean(navigator.mediaDevices?.getUserMedia),
  };
}

function formatBrowserMicStatus(status: BrowserMicStatus) {
  return [
    `origin=${status.origin}`,
    `protocol=${status.protocol}`,
    `secure=${status.secureContext}`,
    `mediaDevices=${status.hasMediaDevices}`,
    `getUserMedia=${status.hasGetUserMedia}`,
  ].join(" ");
}

function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [provider, setProvider] = useState("nvidia");
  const [modelsByProvider, setModelsByProvider] = useState<Record<string, string>>({});
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState("");
  const [renamingChatId, setRenamingChatId] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const [level, setLevel] = useState(0);
  const [connected, setConnected] = useState(false);
  const [voiceSaved, setVoiceSaved] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [typedMessage, setTypedMessage] = useState("");
  const [promptSaved, setPromptSaved] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState("");
  const [pendingAudioUrl, setPendingAudioUrl] = useState("");
  const [queuedAudioCount, setQueuedAudioCount] = useState(0);
  const [audioPlayBlocked, setAudioPlayBlocked] = useState(false);
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("chat");
  const [composerExpanded, setComposerExpanded] = useState(false);
  const [partialTranscript, setPartialTranscript] = useState("");
  const [responseActivity, setResponseActivity] = useState<ResponseActivity>({
    thinking: false,
    synthesizing: false,
    speaking: false,
  });
  const [micStatus, setMicStatus] = useState<BrowserMicStatus>(() => getBrowserMicStatus());
  const socketRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const startingRecordingRef = useRef(false);
  const discardRecordingRef = useRef(false);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioQueueRef = useRef<string[]>([]);
  const audioPlayingRef = useRef(false);
  const audioUnlockedRef = useRef(false);
  const preloadedAudioUrlsRef = useRef<Set<string>>(new Set());
  const textInputRef = useRef<HTMLTextAreaElement | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const liveTranscriptSequenceRef = useRef(0);
  const liveTranscriptLastSentRef = useRef(0);

  useEffect(() => {
    setMicStatus(getBrowserMicStatus());
    fetch("/api/config")
      .then((res) => res.json())
      .then((data: ConfigResponse) => {
        setConfig(data);
        setProvider(data.default_provider);
        setModelsByProvider(defaultModels(data));
        setVoiceSaved(Boolean(data.tts.voice_prompt_saved));
        setSystemPrompt(data.system_prompt);
      })
      .catch((error) => addMessage("system", `Could not load configuration: ${error.message}`, true));
  }, []);

  useEffect(() => {
    void refreshChats();
  }, []);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/chat`);
    socketRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => {
      setPhase("error");
      addMessage("system", "WebSocket connection failed.", true);
    };
    ws.onmessage = (event) => handleSocketMessage(JSON.parse(event.data));
    return () => ws.close();
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const transcriptElement = transcriptRef.current;
      if (!transcriptElement) return;
      transcriptElement.scrollTo({
        top: transcriptElement.scrollHeight,
        behavior: "smooth",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [transcript.length, activeChatId]);

  const configuredProvider = config?.providers.providers[provider];
  const selectedModel = modelsByProvider[provider] || configuredProvider?.model || "";
  const availableModels = configuredProvider?.available_models ?? [];
  const providerOptions = useMemo(() => {
    return Object.entries(config?.providers.providers ?? {});
  }, [config]);
  const visiblePhase = prioritizedPhase(phase, responseActivity);
  const currentPhase = phaseDetail[visiblePhase];
  const activityLevel = visiblePhase === "listening" || visiblePhase === "speaking" ? level : visiblePhase === "idle" || visiblePhase === "error" ? 0 : 0.42;
  const turnInProgress = visiblePhase !== "idle" && visiblePhase !== "error";
  const primaryActionLabel: Record<Phase, string> = {
    idle: "Start recording",
    listening: "Stop",
    transcribing: "Stop",
    thinking: "Stop",
    synthesizing: "Stop",
    speaking: "Stop",
    error: "Start recording",
  };

  async function startRecording() {
    if (startingRecordingRef.current) return;
    if (phase !== "idle" && phase !== "error") return;
    if (recorderRef.current?.state === "recording") return;
    startingRecordingRef.current = true;
    const browserMicStatus = getBrowserMicStatus();
    setMicStatus(browserMicStatus);
    if (!navigator.mediaDevices?.getUserMedia) {
      startingRecordingRef.current = false;
      setPhase("error");
      addMessage(
        "system",
        `Microphone capture is unavailable in this browser context. ${formatBrowserMicStatus(browserMicStatus)}`,
        true,
      );
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      startingRecordingRef.current = false;
      setPhase("error");
      addMessage("system", `Microphone access failed: ${error instanceof Error ? error.message : String(error)}`, true);
      return;
    }
    streamRef.current = stream;
    chunksRef.current = [];
    discardRecordingRef.current = false;
    const mimeType = pickMimeType();
    try {
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
          void maybeSendLiveTranscript();
        }
      };
      recorder.onstop = submitRecording;
      recorderRef.current = recorder;
      startMeter(stream);
      recorder.start(100);
      setPartialTranscript("");
      liveTranscriptSequenceRef.current = 0;
      liveTranscriptLastSentRef.current = 0;
      setPhase("listening");
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      setPhase("error");
      addMessage("system", `Recording could not start: ${error instanceof Error ? error.message : String(error)}`, true);
    } finally {
      startingRecordingRef.current = false;
    }
  }

  function stopRecording(options: { submit: boolean } = { submit: true }) {
    discardRecordingRef.current = !options.submit;
    const wasRecording = recorderRef.current?.state === "recording";
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    stopMeter();
    if (wasRecording) setPhase(options.submit ? "transcribing" : "idle");
  }

  function toggleRecording() {
    if (phase === "listening") {
      stopRecording({ submit: true });
      return;
    }
    void startRecording();
  }

  function handlePrimaryAction() {
    void unlockAssistantAudio();
    if (isAssistantAudioPlaying()) {
      cancelCurrentWork();
      return;
    }
    if (phase === "idle" || phase === "error" || phase === "listening") {
      toggleRecording();
      return;
    }
    cancelCurrentWork();
  }

  async function submitRecording() {
    if (discardRecordingRef.current) {
      chunksRef.current = [];
      discardRecordingRef.current = false;
      return;
    }
    const blob = new Blob(chunksRef.current, { type: recorderRef.current?.mimeType || "audio/webm" });
    const audio = await blobToBase64(blob);
    socketRef.current?.send(
      JSON.stringify({ type: "turn", audio, mimeType: blob.type, provider, model: selectedModel, chatId: activeChatId })
    );
  }

  async function maybeSendLiveTranscript() {
    const recorder = recorderRef.current;
    const socket = socketRef.current;
    if (!recorder || recorder.state !== "recording" || !socket || socket.readyState !== WebSocket.OPEN) return;
    const now = Date.now();
    if (now - liveTranscriptLastSentRef.current < 1800 || chunksRef.current.length < 4) return;
    liveTranscriptLastSentRef.current = now;
    const sequence = ++liveTranscriptSequenceRef.current;
    const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
    const audio = await blobToBase64(blob);
    socket.send(JSON.stringify({ type: "live_transcript", audio, mimeType: blob.type, sequence }));
  }

  function cancelCurrentWork() {
    if (isAssistantAudioPlaying()) {
      stopAssistantAudio();
      audioQueueRef.current = [];
      addMessage("system", "Playback stopped.", true);
      return;
    }
    if (phase === "listening") {
      stopRecording({ submit: false });
      addMessage("system", "Recording discarded.", true);
      return;
    }
    if (phase === "speaking") {
      stopAssistantAudio();
      audioQueueRef.current = [];
      addMessage("system", "Playback stopped.", true);
      return;
    }
    if (phase === "transcribing" || phase === "thinking" || phase === "synthesizing") {
      socketRef.current?.send(JSON.stringify({ type: "cancel" }));
      audioQueueRef.current = [];
      setPhase("idle");
      setLevel(0);
      addMessage("system", "Stopped the current turn.", true);
    }
  }

  function sendTypedMessage() {
    void unlockAssistantAudio();
    const text = typedMessage.trim();
    if (!text || turnInProgress || !connected) return;
    setTypedMessage("");
    setComposerExpanded(false);
    textInputRef.current?.blur();
    socketRef.current?.send(
      JSON.stringify({
        type: "text_turn",
        text,
        provider,
        model: selectedModel,
        chatId: activeChatId,
      }),
    );
    setResponseActivity((current) => ({ ...current, thinking: true }));
    setPhase("thinking");
  }

  async function copyMessageText(item: TranscriptItem) {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(item.text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = item.text;
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      setCopiedMessageId(item.id);
      window.setTimeout(() => setCopiedMessageId((current) => (current === item.id ? "" : current)), 1400);
    } catch (error) {
      addMessage("system", `Copy failed: ${error instanceof Error ? error.message : String(error)}`, true);
    }
  }

  async function uploadVoicePrompt(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/voice-prompt", { method: "POST", body: form });
    if (!response.ok) {
      addMessage("system", "Voice prompt upload failed.", true);
      return;
    }
    setVoiceSaved(true);
  }

  async function saveSystemPrompt() {
    setPromptSaved(false);
    const response = await fetch("/api/system-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ system_prompt: systemPrompt })
    });
    if (!response.ok) {
      addMessage("system", "System prompt could not be saved.", true);
      return;
    }
    setPromptSaved(true);
    window.setTimeout(() => setPromptSaved(false), 1800);
  }

  function setProviderModel(model: string) {
    setModelsByProvider((current) => ({ ...current, [provider]: model }));
  }

  function handleSocketMessage(message: any) {
    if (message.type === "status") {
      if (message.phase === "thinking") {
        setResponseActivity((current) => ({ ...current, thinking: true }));
      }
      if (message.phase === "synthesizing") {
        setResponseActivity((current) => ({ ...current, synthesizing: true }));
      }
      if (message.phase === "idle") {
        setResponseActivity((current) => ({ ...current, thinking: false, synthesizing: false }));
      }
      if (message.phase === "error") {
        setResponseActivity({ thinking: false, synthesizing: false, speaking: false });
      }
      setPhase(message.phase);
    }
    if (message.type === "transcript") {
      if (message.role === "user") setPartialTranscript("");
      addMessage(message.role, message.text, false, message.provider, message.model);
    }
    if (message.type === "partial_transcript") {
      setPartialTranscript(message.text);
    }
    if (message.type === "assistant_start") {
      setResponseActivity((current) => ({ ...current, thinking: true }));
      addMessage("assistant", "", false, message.provider, message.model, message.id);
    }
    if (message.type === "assistant_delta") {
      appendMessageText(message.id, message.text);
    }
    if (message.type === "assistant_done") {
      setPartialTranscript("");
      setResponseActivity((current) => ({ ...current, thinking: false, synthesizing: false }));
    }
    if (message.type === "audio") {
      enqueueAssistantAudio(message.url);
    }
    if (message.type === "error") {
      setPhase("error");
      setResponseActivity({ thinking: false, synthesizing: false, speaking: false });
      addMessage("system", message.message, true);
    }
    if (message.type === "cancelled") {
      setPhase("idle");
      setResponseActivity({ thinking: false, synthesizing: false, speaking: false });
      setLevel(0);
    }
    if (message.type === "chat") {
      if (message.chat) {
        setActiveChatId(message.chat.id);
        setTranscript(toTranscript(message.chat.messages ?? []));
      }
      if (message.chats) setChats(message.chats);
    }
  }

  function playAssistantAudio(url: string) {
    const audio = audioRef.current ?? new Audio();
    audioRef.current = audio;
    audioPlayingRef.current = true;
    audio.preload = "auto";
    audio.setAttribute("playsinline", "true");
    audio.src = url;
    setPendingAudioUrl(url);
    setAudioPlayBlocked(false);
    audio.onplay = () => {
      setAudioPlayBlocked(false);
      setResponseActivity((current) => ({ ...current, speaking: true }));
      pulsePlayback();
    };
    audio.onended = () => {
      setLevel(0);
      setPendingAudioUrl("");
      audioPlayingRef.current = false;
      setResponseActivity((current) => ({ ...current, speaking: false }));
      playNextAssistantAudio();
    };
    audio.onerror = () => {
      setLevel(0);
      setPendingAudioUrl("");
      audioPlayingRef.current = false;
      setResponseActivity((current) => ({ ...current, speaking: false }));
      addMessage("system", "Skipped one speech chunk the browser could not play.", true);
      playNextAssistantAudio({ ignoreBlocked: true });
    };
    void audio.play().catch((error) => {
      setPhase("idle");
      setLevel(0);
      setAudioPlayBlocked(true);
      audioPlayingRef.current = false;
      setResponseActivity((current) => ({ ...current, speaking: false }));
      addMessage("system", `Audio is ready. Tap Play audio to hear it. ${error instanceof Error ? error.message : ""}`.trim(), true);
    });
  }

  function enqueueAssistantAudio(url: string) {
    const queuedUrl = `${url}?t=${Date.now()}`;
    audioQueueRef.current.push(queuedUrl);
    preloadAudioUrl(queuedUrl);
    setQueuedAudioCount(audioQueueRef.current.length);
    playNextAssistantAudio();
  }

  function playNextAssistantAudio(options: { ignoreBlocked?: boolean } = {}) {
    if (audioPlayingRef.current || (audioPlayBlocked && !options.ignoreBlocked)) return;
    const nextAudio = audioQueueRef.current.shift();
    setQueuedAudioCount(audioQueueRef.current.length);
    if (!nextAudio) {
      setPhase((current) => (current === "speaking" || current === "synthesizing" ? "idle" : current));
      return;
    }
    const nextQueuedAudio = audioQueueRef.current[0];
    if (nextQueuedAudio) preloadAudioUrl(nextQueuedAudio);
    playAssistantAudio(nextAudio);
  }

  function preloadAudioUrl(url: string) {
    if (preloadedAudioUrlsRef.current.has(url)) return;
    preloadedAudioUrlsRef.current.add(url);
    void fetch(url, { cache: "force-cache" })
      .then((response) => {
        if (!response.ok) throw new Error(`Audio preload failed: ${response.status}`);
        return response.arrayBuffer();
      })
      .catch(() => {
        preloadedAudioUrlsRef.current.delete(url);
      });
  }

  async function resumeAssistantAudio() {
    const audio = audioRef.current;
    await unlockAssistantAudio();
    setAudioPlayBlocked(false);
    if (audio && audio.src && audio.currentSrc !== SILENT_AUDIO_SRC && audio.paused && !audio.ended) {
      audioPlayingRef.current = true;
      void audio.play().catch((error) => {
        audioPlayingRef.current = false;
        setAudioPlayBlocked(true);
        addMessage("system", `Audio playback failed: ${error instanceof Error ? error.message : String(error)}`, true);
      });
      return;
    }
    playNextAssistantAudio({ ignoreBlocked: true });
  }

  async function unlockAssistantAudio() {
    if (audioUnlockedRef.current) return true;
    const audio = audioRef.current ?? new Audio();
    audioRef.current = audio;
    const previousSrc = audio.currentSrc || audio.src;
    const previousVolume = audio.volume;
    const previousMuted = audio.muted;
    try {
      audio.setAttribute("playsinline", "true");
      audio.preload = "auto";
      audio.volume = 0;
      audio.muted = true;
      audio.src = SILENT_AUDIO_SRC;
      await audio.play();
      audio.pause();
      audio.currentTime = 0;
      audioUnlockedRef.current = true;
      return true;
    } catch {
      audioUnlockedRef.current = false;
      return false;
    } finally {
      audio.volume = previousVolume;
      audio.muted = previousMuted;
      if (previousSrc) {
        audio.src = previousSrc;
      } else {
        audio.removeAttribute("src");
        audio.load();
      }
    }
  }

  function isAssistantAudioPlaying() {
    return audioPlayingRef.current;
  }

  function stopAssistantAudio() {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.removeAttribute("src");
      audio.load();
    }
    stopMeter();
    audioPlayingRef.current = false;
    audioQueueRef.current = [];
    preloadedAudioUrlsRef.current.clear();
    setQueuedAudioCount(0);
    setResponseActivity({ thinking: false, synthesizing: false, speaking: false });
    setLevel(0);
    setPhase("idle");
    setPendingAudioUrl("");
    setAudioPlayBlocked(false);
  }

  function addMessage(role: TranscriptItem["role"], text: string, error = false, itemProvider?: string, model?: string | null, id?: string) {
    setTranscript((items) => [
      ...items,
      {
        id: id || makeId(),
        role,
        text,
        error,
        provider: itemProvider,
        model,
        time: new Intl.DateTimeFormat([], { hour: "numeric", minute: "2-digit", second: "2-digit" }).format(new Date())
      }
    ]);
  }

  function appendMessageText(id: string, text: string) {
    setTranscript((items) =>
      items.map((item) => (item.id === id ? { ...item, text: `${item.text}${text}` } : item)),
    );
  }

  async function refreshChats() {
    const response = await fetch("/api/chats");
    const data = await response.json();
    setChats(data.chats ?? []);
    const nextActive = data.active_chat?.id || activeChatId || data.chats?.[0]?.id;
    if (nextActive) await loadChat(nextActive);
  }

  async function loadChat(chatId: string) {
    const response = await fetch(`/api/chats/${chatId}`);
    if (!response.ok) return;
    const data = await response.json();
    setActiveChatId(data.chat.id);
    setTranscript(toTranscript(data.chat.messages ?? []));
    setMobilePanel("chat");
  }

  async function createChat() {
    const response = await fetch("/api/chats", { method: "POST" });
    if (!response.ok) return;
    const data = await response.json();
    setChats(data.chats ?? []);
    setActiveChatId(data.chat.id);
    setTranscript([]);
    setMobilePanel("chat");
  }

  async function updateChat(chatId: string, patch: { title?: string; pinned?: boolean }) {
    const response = await fetch(`/api/chats/${chatId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch)
    });
    if (!response.ok) return;
    const data = await response.json();
    setChats(data.chats ?? []);
    if (data.chat.id === activeChatId) setTranscript(toTranscript(data.chat.messages ?? transcript));
  }

  async function deleteChat(chatId: string) {
    const response = await fetch(`/api/chats/${chatId}`, { method: "DELETE" });
    if (!response.ok) return;
    const data = await response.json();
    setChats(data.chats ?? []);
    const nextId = data.active_chat?.id || data.chats?.[0]?.id || "";
    if (chatId === activeChatId && nextId) {
      await loadChat(nextId);
    }
  }

  function startRename(chat: ChatSummary) {
    setRenamingChatId(chat.id);
    setRenameValue(chat.title);
  }

  async function finishRename(chatId: string) {
    await updateChat(chatId, { title: renameValue });
    setRenamingChatId("");
    setRenameValue("");
  }

  function startMeter(stream: MediaStream) {
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    audioContextRef.current = context;
    analyserRef.current = analyser;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const draw = () => {
      analyser.getByteFrequencyData(data);
      const avg = data.reduce((sum, value) => sum + value, 0) / data.length;
      setLevel(Math.min(1, avg / 90));
      animationRef.current = requestAnimationFrame(draw);
    };
    draw();
  }

  function stopMeter() {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    void audioContextRef.current?.close();
    analyserRef.current = null;
  }

  function pulsePlayback() {
    let tick = 0;
    const animate = () => {
      if (audioRef.current?.paused) return;
      tick += 0.12;
      setLevel(0.35 + Math.abs(Math.sin(tick)) * 0.55);
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();
  }

  return (
    <main className={`shell phase-${visiblePhase} mobile-panel-${mobilePanel}`}>
      <section className="topbar">
        <div>
          <p className="eyebrow">AI live chat</p>
          <h1>Rick</h1>
        </div>
        <div className={connected ? "connection online" : "connection offline"}>
          {connected ? <Wifi size={18} /> : <WifiOff size={18} />}
          {connected ? "Connected" : "Offline"}
        </div>
      </section>

      <nav className="mobile-tabs" aria-label="Mobile sections">
        <button type="button" className={mobilePanel === "chat" ? "active" : ""} onClick={() => setMobilePanel("chat")} aria-pressed={mobilePanel === "chat"}>
          <MessageCircle size={17} />
          Live Chat
        </button>
        <button type="button" className={mobilePanel === "chats" ? "active" : ""} onClick={() => setMobilePanel("chats")} aria-pressed={mobilePanel === "chats"}>
          <MessagesSquare size={17} />
          Chats
        </button>
        <button type="button" className={mobilePanel === "settings" ? "active" : ""} onClick={() => setMobilePanel("settings")} aria-pressed={mobilePanel === "settings"}>
          <SlidersHorizontal size={17} />
          Settings
        </button>
      </nav>

      <section className="workspace">
        <aside className="chat-list">
          <button className="new-chat" onClick={createChat}>
            <Plus size={18} />
            New chat
          </button>
          <div className="chat-scroll">
            {chats.map((chat) => (
              <div key={chat.id} className={`chat-row ${chat.id === activeChatId ? "selected" : ""}`}>
                <button className="chat-title" onClick={() => loadChat(chat.id)}>
                  {renamingChatId === chat.id ? (
                    <input
                      value={renameValue}
                      autoFocus
                      onChange={(event) => setRenameValue(event.target.value)}
                      onBlur={() => finishRename(chat.id)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") void finishRename(chat.id);
                        if (event.key === "Escape") setRenamingChatId("");
                      }}
                    />
                  ) : (
                    <>
                      <span>{chat.title}</span>
                      <small>{chat.message_count} messages</small>
                    </>
                  )}
                </button>
                <div className="chat-actions">
                  <button title={chat.pinned ? "Unpin chat" : "Pin chat"} onClick={() => updateChat(chat.id, { pinned: !chat.pinned })}>
                    {chat.pinned ? <PinOff size={15} /> : <Pin size={15} />}
                  </button>
                  <button title="Rename chat" onClick={() => startRename(chat)}>
                    <Pencil size={15} />
                  </button>
                  <button title="Delete chat" onClick={() => deleteChat(chat.id)}>
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </aside>
        <aside className="controls">
          <section className="voice-stage" aria-live="polite">
            <div className="phase-chip">
              {currentPhase.icon}
              <span>{currentPhase.label}</span>
            </div>
            <div className="meter" style={{ "--level": activityLevel } as React.CSSProperties}>
              <div className="halo halo-one" />
              <div className="halo halo-two" />
              <div className="halo halo-three" />
              <div className="core">
                {visiblePhase === "listening" ? <Mic size={42} /> : visiblePhase === "speaking" ? <Volume2 size={42} /> : currentPhase.icon}
              </div>
              <div className="waveform" aria-hidden="true">
                {Array.from({ length: 18 }, (_, index) => (
                  <span key={index} style={{ "--bar": index } as React.CSSProperties} />
                ))}
              </div>
            </div>
            <div className="phase-copy">
              <strong>{phaseLabel[visiblePhase]}</strong>
              <span>{currentPhase.detail}</span>
            </div>
          </section>
          <button
            className={`record ${phase !== "idle" && phase !== "error" ? "active" : ""}`}
            onClick={handlePrimaryAction}
            disabled={!connected}
          >
            {visiblePhase === "idle" || visiblePhase === "error" ? <Mic size={20} /> : <Square size={20} />}
            {primaryActionLabel[visiblePhase]}
          </button>

          <section className="control-group">
            <label className="upload">
              <Upload size={18} />
              <span>{voiceSaved ? "Voice template saved" : "Upload voice MP3"}</span>
              <input type="file" accept="audio/*,.mp3" onChange={uploadVoicePrompt} />
            </label>

            <label className="field">
              <span>Provider</span>
              <select value={provider} onChange={(event) => setProvider(event.target.value)} disabled={turnInProgress}>
                {providerOptions.map(([key, item]) => (
                  <option key={key} value={key} disabled={item.disabled}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Model</span>
              {availableModels.length > 0 ? (
                <select value={selectedModel} onChange={(event) => setProviderModel(event.target.value)} disabled={turnInProgress}>
                  {selectedModel && !availableModels.includes(selectedModel) ? <option value={selectedModel}>{selectedModel}</option> : null}
                  {availableModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              ) : (
                <input value={selectedModel} onChange={(event) => setProviderModel(event.target.value)} placeholder="Model id" disabled={turnInProgress} />
              )}
            </label>
          </section>

          <section className="control-group">
            <label className="field prompt-field">
              <span>Personality and memory</span>
              <textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} />
            </label>
            <button className="secondary" onClick={saveSystemPrompt}>
              {promptSaved ? "Saved" : "Save prompt"}
            </button>
          </section>

          <section className="status-grid">
            <div className={configuredProvider?.configured ? "status ok" : "status warn"}>
              <strong>{configuredProvider?.name ?? "Provider"}</strong>
              <span>{providerDetail(configuredProvider)}</span>
            </div>

            <div className={micStatus.hasGetUserMedia ? "status ok" : "status warn"}>
              <strong>Microphone</strong>
              <span>{micStatus.hasGetUserMedia ? "Secure browser capture is available." : formatBrowserMicStatus(micStatus)}</span>
            </div>
          </section>
        </aside>

        <section className={`chat-pane ${composerExpanded ? "composer-expanded" : ""}`}>
          <audio ref={audioRef} preload="auto" playsInline className="assistant-audio" />
          <div className="transcript" ref={transcriptRef}>
            {transcript.length === 0 && !partialTranscript ? (
              <div className="empty">
                <h2>Start recording or type a message.</h2>
                <p>Replies will appear here with provider, model, timestamps, and playback.</p>
              </div>
            ) : (
              <>
                {transcript.map((item) => (
                  <MessageArticle key={item.id} item={item} copiedMessageId={copiedMessageId} copyMessageText={copyMessageText} />
                ))}
                {partialTranscript ? (
                  <article className="message user partial">
                    <header>
                      <span>You</span>
                      <div className="message-tools">
                        <time>Live</time>
                      </div>
                    </header>
                    <p>{partialTranscript}</p>
                  </article>
                ) : null}
              </>
            )}
          </div>
          {audioPlayBlocked && (pendingAudioUrl || queuedAudioCount > 0) ? (
            <div className="playback-fallback">
              <span>Audio is ready.</span>
              <button type="button" onClick={resumeAssistantAudio}>
                <Play size={16} />
                Play audio
              </button>
            </div>
          ) : null}
          <form
            className={`text-composer ${composerExpanded ? "expanded" : ""}`}
            onSubmit={(event) => {
              event.preventDefault();
              sendTypedMessage();
            }}
          >
            <textarea
              ref={textInputRef}
              value={typedMessage}
              onChange={(event) => setTypedMessage(event.target.value)}
              onFocus={() => setComposerExpanded(true)}
              onBlur={() => {
                if (!typedMessage.trim()) setComposerExpanded(false);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendTypedMessage();
                }
              }}
              placeholder="Type a message"
              rows={2}
              disabled={!connected || turnInProgress}
            />
            <button type="submit" disabled={!connected || turnInProgress || !typedMessage.trim()} title="Send message">
              <Send size={18} />
              Send
            </button>
          </form>
        </section>
      </section>
    </main>
  );
}

function defaultModels(config: ConfigResponse) {
  const result: Record<string, string> = {};
  Object.entries(config.providers.providers).forEach(([key, item]) => {
    result[key] = item.model || item.available_models?.[0] || "";
  });
  return result;
}

function toTranscript(messages: any[]): TranscriptItem[] {
  return messages.map((message) => ({
    id: message.id ?? makeId(),
    role: message.role,
    text: message.text,
    provider: message.provider,
    model: message.model,
    error: Boolean(message.error),
    time: formatTime(message.time)
  }));
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat([], { hour: "numeric", minute: "2-digit", second: "2-digit" }).format(date);
}

function providerDetail(provider?: ProviderStatus) {
  if (!provider) return "Configuration unavailable.";
  if (provider.disabled) return provider.reason ?? "Disabled.";
  if (provider.configured) return provider.model ? `Model: ${provider.model}` : "Configured.";
  return `Missing ${provider.missing?.join(", ") || "configuration"}.`;
}

function prioritizedPhase(phase: Phase, activity: ResponseActivity): Phase {
  if (activity.speaking) return "speaking";
  if (activity.thinking) return "thinking";
  if (activity.synthesizing) return "synthesizing";
  return phase;
}

function MessageArticle({
  item,
  copiedMessageId,
  copyMessageText,
}: {
  item: TranscriptItem;
  copiedMessageId: string;
  copyMessageText: (item: TranscriptItem) => void;
}) {
  return (
    <article className={`message ${item.role} ${item.error ? "error" : ""}`}>
      <header>
        <span>{item.role === "assistant" ? "Assistant" : item.role === "user" ? "You" : "System"}</span>
        <div className="message-tools">
          <button type="button" onClick={() => copyMessageText(item)} title="Copy message text">
            <Copy size={14} />
            {copiedMessageId === item.id ? "Copied" : "Copy"}
          </button>
          <time>{item.time}</time>
        </div>
      </header>
      <p>{item.text}</p>
      {item.provider ? <footer>{item.provider}{item.model ? ` / ${item.model}` : ""}</footer> : null}
    </article>
  );
}

function pickMimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

createRoot(document.getElementById("root")!).render(<App />);
