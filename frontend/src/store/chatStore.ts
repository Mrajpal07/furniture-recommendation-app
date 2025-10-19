import { create } from 'zustand'
import type { ChatMessage } from '../lib/types'

interface ChatStore {
    isOpen: boolean
    messages: ChatMessage[]
    sessionId: string
    isLoading: boolean
    
    openChat: () => void
    closeChat: () => void
    toggleChat: () => void
    addMessage: (message: ChatMessage) => void
    setMessages: (messages: ChatMessage[]) => void
    setLoading: (loading: boolean) => void
    clearMessages: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
    isOpen: false,
    messages: [],
    sessionId: `session_${Date.now()}`,
    isLoading: false,
    
    openChat: () => set({ isOpen: true }),
    closeChat: () => set({ isOpen: false }),
    toggleChat: () => set((state) => ({ isOpen: !state.isOpen })),
    
    addMessage: (message) => set((state) => ({
        messages: [...state.messages, message]
    })),
    
    setMessages: (messages) => set({ messages }),
    setLoading: (loading) => set({ isLoading: loading }),
    
    clearMessages: () => set({
    messages: [],
    sessionId: `session_${Date.now()}`
    }),
}))