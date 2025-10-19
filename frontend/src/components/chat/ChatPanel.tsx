import { useState, useRef, useEffect } from 'react'
import { X, Send, Sparkles, Loader2, RotateCcw, Bot } from 'lucide-react'
import { useChatStore } from '../../store/chatStore'
import { chatApi } from '../../lib/api'
import { MessageBubble } from './MessageBubble'






export function ChatPanel() {
    const { isOpen, closeChat, messages, addMessage, isLoading, setLoading, sessionId, clearMessages } = useChatStore()
    const [input, setInput] = useState('')
    const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
    useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = {
        role: 'user' as const,
        content: input,
        timestamp: new Date().toISOString()
    }

    addMessage(userMessage)
    setInput('')
    setLoading(true)

    try {
        const response = await chatApi.sendMessage({
        message: input,
        session_id: sessionId
        })

        const assistantMessage = {
        role: 'assistant' as const,
        content: response.message,
        timestamp: new Date().toISOString(),
        products: response.products
        }

        addMessage(assistantMessage)
    } catch (error) {
        console.error('Chat error:', error)
        addMessage({
        role: 'assistant' as const,
        content: "Sorry, I'm having trouble connecting. Please try again!",
        timestamp: new Date().toISOString()
        })
    } finally {
        setLoading(false)
    }
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
    }
    }

    if (!isOpen) return null

    return (
    <>
      {/* Overlay */}
        <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 animate-in fade-in duration-200"
        onClick={closeChat}
        />

      {/* Chat Panel */}
        <div className="fixed right-0 top-0 h-full w-full sm:w-[480px] bg-background border-l border-border z-50 flex flex-col animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border p-4 bg-gradient-to-r from-blue-600/10 to-indigo-600/10">
            <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg">
                <Sparkles className="h-5 w-5" />
            </div>
            <div>
                <h2 className="font-semibold text-foreground">AI Assistant</h2>
                <p className="text-xs text-muted-foreground">Powered by Groq</p>
            </div>
            </div>

            <div className="flex items-center gap-2">
            <button
                onClick={clearMessages}
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-border hover:bg-accent transition-colors"
                title="Clear conversation"
            >
                <RotateCcw className="h-4 w-4" />
            </button>
            <button
                onClick={closeChat}
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-border hover:bg-accent transition-colors"
            >
                <X className="h-5 w-5" />
            </button>
            </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white">
                <Sparkles className="h-8 w-8" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Start a Conversation</h3>
                <p className="text-sm text-muted-foreground max-w-xs mb-6">
                Ask me anything about furniture! I can help you find products, compare options, or give recommendations.
                </p>
                
              {/* Suggested Prompts */}
                <div className="space-y-2 w-full max-w-sm">
                <p className="text-xs font-medium text-muted-foreground text-left">Try asking:</p>
                {[
                    "Show me modern sofas under $500",
                    "I need furniture for a small bedroom",
                    "What's the difference between these chairs?"
                ].map((suggestion, i) => (
                    <button
                    key={i}
                    onClick={() => setInput(suggestion)}
                    className="w-full text-left rounded-lg border border-border bg-background px-4 py-3 text-sm hover:bg-accent transition-colors"
                    >
                    {suggestion}
                    </button>
                ))}
                </div>
            </div>
            ) : (
            <>
                {messages.map((message, i) => (
                <MessageBubble key={i} message={message} />
                ))}
                {isLoading && (
                <div className="flex gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 text-white">
                    <Bot className="h-4 w-4" />
                </div>
                <div className="flex items-center gap-2 rounded-2xl bg-muted px-4 py-2.5 border border-border">
                    <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                </div>
                </div>
                )}
                <div ref={messagesEndRef} />
            </>
            )}
        </div>

        {/* Input */}
        <div className="border-t border-border p-4 bg-background">
            <div className="flex gap-2">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about furniture..."
                disabled={isLoading}
                className="flex-1 rounded-xl border border-border bg-background px-4 py-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-xl transition-all hover:scale-105 active:scale-95"
            >
                <Send className="h-5 w-5" />
            </button>
            </div>
        </div>
        </div>
    </>
    )
}