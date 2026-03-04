'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Loader2, Activity, Brain, Sparkles, Battery, Settings, X, Key } from 'lucide-react';
import axios from 'axios';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface CompanionState {
  solvency: number;
  energy: number;
  mood: number;
  curiosity: number;
  social_satiation: number;
  confidence: number;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [companionId, setCompanionId] = useState<string | null>(null);
  const [companionState, setCompanionState] = useState<CompanionState | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [hasApiKey, setHasApiKey] = useState(false);
  const [llmProvider, setLlmProvider] = useState<'deepseek' | 'openai' | 'claude'>('deepseek');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const fetchCompanionState = async () => {
    try {
      const response = await axios.get(`/api/companions/${companionId}`);
      setCompanionState(response.data);
    } catch (error) {
      console.log('Companion not found, will be created on first message');
    }
  };

  useEffect(() => {
    fetchCompanionState();
    checkApiKey();
    // Load saved API key from localStorage
    const savedKey = localStorage.getItem('deepseek_api_key');
    if (savedKey) {
      setApiKey(savedKey);
      setHasApiKey(true);
    }
  }, []);

  const checkApiKey = async () => {
    try {
      const response = await axios.get('/');
      setHasApiKey(response.data.llm_configured);
      if (!response.data.llm_configured) {
        setShowSettings(true);
      }
    } catch (error) {
      console.error('Error checking API key status:', error);
    }
  };

  const saveApiKey = async () => {
    try {
      // For now, we'll store it in localStorage and send with each request
      localStorage.setItem('deepseek_api_key', apiKey);
      setHasApiKey(true);
      setShowSettings(false);
    } catch (error) {
      console.error('Error saving API key:', error);
    }
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
      // Don't send companion_id on first message to auto-create
      const payload: any = {
        message: input,
        api_key: storedKey,
      };
      
      // Only include companion_id if we have one
      if (companionId) {
        payload.companion_id = companionId;
      }
      
      const response = await axios.post('/api/chat', payload);

      // Update companion ID from response if this is the first message
      if (!companionId && response.data.companion_id) {
        setCompanionId(response.data.companion_id);
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      
      // Update companion state
      if (response.data.state) {
        setCompanionState(response.data.state);
      } else {
        fetchCompanionState();
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'I encountered an error. Please make sure the API key is configured.',
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
    <div className="flex h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Sidebar with companion state */}
      <div className="w-80 bg-black/30 backdrop-blur-sm border-r border-gray-800 p-6">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Brain className="w-6 h-6 text-purple-400" />
              Frisbot
            </h1>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
              title="Settings"
            >
              <Settings className="w-5 h-5 text-gray-400" />
            </button>
          </div>
          <p className="text-gray-400 text-sm">Bayesian Cognitive Companion</p>
        </div>

        {companionState && (
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Internal States</h2>
            
            <div className="space-y-3">
              <StateIndicator 
                label="Solvency" 
                value={companionState.solvency} 
                icon={<Battery className="w-4 h-4" />}
                color="blue"
              />
              <StateIndicator 
                label="Energy" 
                value={companionState.energy} 
                icon={<Activity className="w-4 h-4" />}
                color="green"
              />
              <StateIndicator 
                label="Mood" 
                value={companionState.mood} 
                icon={<Sparkles className="w-4 h-4" />}
                color="yellow"
              />
              <StateIndicator 
                label="Curiosity" 
                value={companionState.curiosity} 
                icon={<Brain className="w-4 h-4" />}
                color="purple"
              />
              <StateIndicator 
                label="Social" 
                value={companionState.social_satiation} 
                icon={<User className="w-4 h-4" />}
                color="pink"
              />
              <StateIndicator 
                label="Confidence" 
                value={companionState.confidence} 
                icon={<Bot className="w-4 h-4" />}
                color="indigo"
              />
            </div>
          </div>
        )}
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Brain className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                <h2 className="text-xl text-gray-300 mb-2">Start a conversation with Frisbot</h2>
                <p className="text-gray-500">I maintain uncertain beliefs about my internal states</p>
              </div>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-purple-400" />
                </div>
              )}
              
              <div
                className={`max-w-2xl p-4 rounded-2xl ${
                  message.role === 'user'
                    ? 'bg-blue-600/20 text-blue-100 border border-blue-500/30'
                    : 'bg-gray-800/50 text-gray-100 border border-gray-700/50'
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
          ))}
          
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                <Bot className="w-5 h-5 text-purple-400" />
              </div>
              <div className="bg-gray-800/50 text-gray-100 p-4 rounded-2xl border border-gray-700/50">
                <Loader2 className="w-5 h-5 animate-spin" />
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="border-t border-gray-800 p-6">
          <div className="flex gap-3 max-w-4xl mx-auto">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="flex-1 bg-gray-800/50 text-white px-4 py-3 rounded-xl border border-gray-700 focus:outline-none focus:border-purple-500 transition-colors"
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
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Key className="w-5 h-5 text-purple-400" />
                API Configuration
              </h2>
              <button
                onClick={() => setShowSettings(false)}
                className="p-1 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  DeepSeek API Key
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-gray-800 text-white px-4 py-2 rounded-lg border border-gray-700 focus:outline-none focus:border-purple-500 transition-colors"
                />
                <p className="text-xs text-gray-500 mt-2">
                  Get your API key from{' '}
                  <a
                    href="https://platform.deepseek.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:underline"
                  >
                    platform.deepseek.com
                  </a>
                </p>
              </div>

              {!hasApiKey && (
                <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg p-3">
                  <p className="text-sm text-yellow-200">
                    ⚠️ No API key configured. The chat will not work without a valid DeepSeek API key.
                  </p>
                </div>
              )}

              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setShowSettings(false)}
                  className="flex-1 bg-gray-800 hover:bg-gray-700 text-white py-2 px-4 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={saveApiKey}
                  disabled={!apiKey.trim()}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white py-2 px-4 rounded-lg transition-colors"
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

function StateIndicator({ 
  label, 
  value, 
  icon, 
  color 
}: { 
  label: string; 
  value: number; 
  icon: React.ReactNode;
  color: string;
}) {
  const percentage = Math.round(value * 100);
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    purple: 'bg-purple-500',
    pink: 'bg-pink-500',
    indigo: 'bg-indigo-500',
  };
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-400 flex items-center gap-2">
          {icon}
          {label}
        </span>
        <span className="text-gray-300">{percentage}%</span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-2">
        <div 
          className={`h-2 rounded-full transition-all duration-500 ${colorClasses[color as keyof typeof colorClasses]}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}