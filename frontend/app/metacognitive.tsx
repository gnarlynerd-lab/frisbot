'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Loader2, Activity, Brain, Sparkles, Battery, Settings, X, Key, Eye, Zap, TrendingUp, AlertCircle } from 'lucide-react';
import axios from 'axios';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  analysis?: any;
  metacognitive?: any;
}

interface CognitiveState {
  solvency: number;
  energy: { value: number; uncertainty: number };
  mood: { value: number; uncertainty: number };
  curiosity: { value: number; uncertainty: number };
  social_satiation: { value: number; uncertainty: number };
  confidence: { value: number; uncertainty: number };
}

interface MetacognitiveState {
  metacognitive_beliefs: {
    understanding_accuracy: { mu: number; sigma: number; precision: number };
    response_coherence: { mu: number; sigma: number; precision: number };
    cognitive_load: { mu: number; sigma: number; precision: number };
    epistemic_confidence: { mu: number; sigma: number; precision: number };
    model_uncertainty: { mu: number; sigma: number; precision: number };
  };
  overall_metacognitive_confidence: number;
  average_metacognitive_uncertainty: number;
  cognitive_load: number;
  model_uncertainty: number;
  solvency: number;
}

export default function MetacognitiveChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [companionId, setCompanionId] = useState<string | null>(null);
  const [cognitiveState, setCognitiveState] = useState<CognitiveState | null>(null);
  const [metacognitiveState, setMetacognitiveState] = useState<MetacognitiveState | null>(null);
  const [cognitiveCoherence, setCognitiveCoherence] = useState<number>(0.5);
  const [showSettings, setShowSettings] = useState(false);
  const [showMetacognitive, setShowMetacognitive] = useState(true);
  const [apiKey, setApiKey] = useState('');
  const [enableIntrospection, setEnableIntrospection] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load saved API key
    const savedKey = localStorage.getItem('deepseek_api_key');
    if (savedKey) {
      setApiKey(savedKey);
    } else {
      setShowSettings(true);
    }
  }, []);

  const saveApiKey = () => {
    localStorage.setItem('deepseek_api_key', apiKey);
    setShowSettings(false);
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const storedKey = localStorage.getItem('deepseek_api_key');
      const payload: any = {
        message: input,
        api_key: storedKey,
        enable_deep_introspection: enableIntrospection,
      };
      
      if (companionId) {
        payload.companion_id = companionId;
      }
      
      // Use metacognitive endpoint on port 8001
      const response = await axios.post('http://localhost:8001/api/metacognitive/chat', payload);

      if (!companionId && response.data.companion_id) {
        setCompanionId(response.data.companion_id);
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
        analysis: response.data.message_analysis,
        metacognitive: response.data.metacognitive_assessment,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setCognitiveState(response.data.cognitive_state);
      setMetacognitiveState(response.data.metacognitive_state);
      setCognitiveCoherence(response.data.cognitive_coherence);
      
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'I encountered an error. Please check the API key and connection.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950">
      {/* Enhanced Sidebar */}
      <div className="w-96 bg-black/40 backdrop-blur-sm border-r border-purple-800/30 p-6 overflow-y-auto">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Brain className="w-6 h-6 text-purple-400 animate-pulse" />
              Metacognitive Frisbot
            </h1>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 hover:bg-purple-900/30 rounded-lg transition-colors"
            >
              <Settings className="w-5 h-5 text-purple-400" />
            </button>
          </div>
          <p className="text-purple-300 text-sm">Self-Aware Bayesian Intelligence</p>
        </div>

        {/* Cognitive Coherence */}
        <div className="mb-6 p-4 bg-purple-900/20 rounded-lg border border-purple-800/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-purple-300 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Cognitive Coherence
            </span>
            <span className="text-white font-bold">
              {Math.round(cognitiveCoherence * 100)}%
            </span>
          </div>
          <div className="w-full bg-purple-950 rounded-full h-3">
            <div 
              className="h-3 rounded-full bg-gradient-to-r from-purple-600 to-purple-400 transition-all duration-1000"
              style={{ width: `${cognitiveCoherence * 100}%` }}
            />
          </div>
          <p className="text-xs text-purple-400 mt-2">
            Alignment between meta & object beliefs
          </p>
        </div>

        {/* Object-Level Beliefs */}
        {cognitiveState && (
          <div className="space-y-4 mb-6">
            <h2 className="text-sm font-semibold text-purple-300 uppercase tracking-wider">
              Object-Level Beliefs
            </h2>
            
            <div className="space-y-3">
              <BeliefIndicator 
                label="Solvency" 
                value={metacognitiveState?.solvency || 0.7}
                uncertainty={0}
                icon={<Battery className="w-4 h-4" />}
                color="blue"
                isResource
              />
              {Object.entries(cognitiveState).map(([key, belief]) => {
                if (key === 'solvency') return null;
                const value = typeof belief === 'object' ? belief.value : belief;
                const uncertainty = typeof belief === 'object' ? belief.uncertainty : 0;
                
                const icons: Record<string, JSX.Element> = {
                  energy: <Activity className="w-4 h-4" />,
                  mood: <Sparkles className="w-4 h-4" />,
                  curiosity: <Brain className="w-4 h-4" />,
                  social_satiation: <User className="w-4 h-4" />,
                  confidence: <Eye className="w-4 h-4" />,
                };
                
                return (
                  <BeliefIndicator
                    key={key}
                    label={key.replace('_', ' ')}
                    value={value}
                    uncertainty={uncertainty}
                    icon={icons[key]}
                    color="purple"
                  />
                );
              })}
            </div>
          </div>
        )}

        {/* Metacognitive Beliefs */}
        {metacognitiveState && showMetacognitive && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-purple-300 uppercase tracking-wider">
                Metacognitive Beliefs
              </h2>
              <button
                onClick={() => setShowMetacognitive(!showMetacognitive)}
                className="text-purple-400 hover:text-purple-300"
              >
                <Eye className="w-4 h-4" />
              </button>
            </div>
            
            <div className="space-y-3">
              {Object.entries(metacognitiveState.metacognitive_beliefs).map(([key, belief]) => (
                <MetaBeliefIndicator
                  key={key}
                  label={key.replace(/_/g, ' ')}
                  mu={belief.mu}
                  sigma={belief.sigma}
                  precision={belief.precision}
                />
              ))}
            </div>

            <div className="p-3 bg-purple-950/40 rounded-lg border border-purple-800/20">
              <div className="text-xs text-purple-400 space-y-1">
                <div className="flex justify-between">
                  <span>Meta Confidence:</span>
                  <span className="text-purple-200">
                    {Math.round(metacognitiveState.overall_metacognitive_confidence * 100)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Uncertainty:</span>
                  <span className="text-purple-200">
                    {metacognitiveState.average_metacognitive_uncertainty.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Introspection Toggle */}
        <div className="mt-6 p-3 bg-purple-900/20 rounded-lg">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={enableIntrospection}
              onChange={(e) => setEnableIntrospection(e.target.checked)}
              className="rounded text-purple-600"
            />
            <span className="text-sm text-purple-300">Enable Deep Introspection</span>
          </label>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Brain className="w-16 h-16 text-purple-400 mx-auto mb-4 animate-pulse" />
                <h2 className="text-xl text-purple-200 mb-2">Metacognitive Conversation</h2>
                <p className="text-purple-400">I maintain beliefs about my own cognitive processes</p>
              </div>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div key={index}>
              <div className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {message.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5 text-purple-400" />
                  </div>
                )}
                
                <div
                  className={`max-w-2xl p-4 rounded-2xl ${
                    message.role === 'user'
                      ? 'bg-blue-600/20 text-blue-100 border border-blue-500/30'
                      : 'bg-purple-900/20 text-purple-100 border border-purple-700/30'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
                
                {message.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5 text-blue-400" />
                  </div>
                )}
              </div>

              {/* Show metacognitive assessment for assistant messages */}
              {message.role === 'assistant' && message.metacognitive && (
                <div className="ml-11 mt-2 p-2 bg-purple-950/30 rounded-lg border border-purple-800/20">
                  <p className="text-xs text-purple-400 mb-1">Self-Assessment:</p>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="text-purple-300">
                      Understanding: {Math.round(message.metacognitive.understanding_confidence * 100)}%
                    </div>
                    <div className="text-purple-300">
                      Processing: {Math.round(message.metacognitive.processing_difficulty * 100)}%
                    </div>
                    <div className="text-purple-300">
                      Uncertainty: {Math.round(message.metacognitive.model_uncertainty * 100)}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                <Bot className="w-5 h-5 text-purple-400 animate-pulse" />
              </div>
              <div className="bg-purple-900/20 text-purple-100 p-4 rounded-2xl border border-purple-700/30">
                <Loader2 className="w-5 h-5 animate-spin" />
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-purple-800/30 p-6 bg-black/20 backdrop-blur-sm">
          <div className="flex gap-3 max-w-4xl mx-auto">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Explore metacognitive awareness..."
              className="flex-1 bg-purple-950/50 text-white px-4 py-3 rounded-xl border border-purple-700/50 focus:outline-none focus:border-purple-500 transition-colors placeholder-purple-400/50"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-purple-950 border border-purple-800 rounded-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Key className="w-5 h-5 text-purple-400" />
                API Configuration
              </h2>
              <button
                onClick={() => setShowSettings(false)}
                className="p-1 hover:bg-purple-900 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-purple-400" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-purple-300 mb-2">
                  DeepSeek API Key
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-purple-900/50 text-white px-4 py-2 rounded-lg border border-purple-700 focus:outline-none focus:border-purple-500 transition-colors"
                />
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setShowSettings(false)}
                  className="flex-1 bg-purple-800 hover:bg-purple-700 text-white py-2 px-4 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={saveApiKey}
                  disabled={!apiKey.trim()}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white py-2 px-4 rounded-lg transition-colors"
                >
                  Save Key
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function BeliefIndicator({ 
  label, 
  value, 
  uncertainty,
  icon, 
  color,
  isResource = false
}: { 
  label: string; 
  value: number; 
  uncertainty: number;
  icon: React.ReactNode;
  color: string;
  isResource?: boolean;
}) {
  const percentage = Math.round(value * 100);
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-purple-300 flex items-center gap-2">
          {icon}
          {label}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-purple-100">{percentage}%</span>
          {uncertainty > 0 && (
            <span className="text-purple-500 text-xs">±{Math.round(uncertainty * 100)}</span>
          )}
        </div>
      </div>
      <div className="w-full bg-purple-950 rounded-full h-2">
        <div 
          className={`h-2 rounded-full transition-all duration-500 ${
            isResource ? 'bg-gradient-to-r from-blue-600 to-blue-400' : 'bg-gradient-to-r from-purple-600 to-purple-400'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function MetaBeliefIndicator({
  label,
  mu,
  sigma,
  precision
}: {
  label: string;
  mu: number;
  sigma: number;
  precision: number;
}) {
  return (
    <div className="p-2 bg-purple-950/30 rounded-lg border border-purple-800/20">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs text-purple-400 capitalize">{label}</span>
        <span className="text-xs text-purple-200">{Math.round(mu * 100)}%</span>
      </div>
      <div className="flex gap-2 text-xs">
        <span className="text-purple-500">σ: {sigma.toFixed(3)}</span>
        <span className="text-purple-500">τ: {precision.toFixed(1)}</span>
      </div>
    </div>
  );
}