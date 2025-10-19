import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ThemeStore {
    isDark: boolean
    toggleTheme: () => void
    setTheme: (isDark: boolean) => void
}

export const useThemeStore = create<ThemeStore>()(
    persist(
    (set) => ({
      isDark: true, // Default to dark mode
        toggleTheme: () => set((state) => {
        const newIsDark = !state.isDark
        // Update document class
        if (newIsDark) {
            document.documentElement.classList.add('dark')
        } else {
            document.documentElement.classList.remove('dark')
        }
        return { isDark: newIsDark }
        }),
        setTheme: (isDark) => set(() => {
        // Update document class
        if (isDark) {
            document.documentElement.classList.add('dark')
        } else {
            document.documentElement.classList.remove('dark')
        }
        return { isDark }
        }),
    }),
    {
        name: 'theme-storage',
    }
    )
)

// Initialize theme on load
if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('theme-storage')
    if (stored) {
    const { state } = JSON.parse(stored)
    if (state.isDark) {
        document.documentElement.classList.add('dark')
    }
    } else {
    // Default to dark
    document.documentElement.classList.add('dark')
    }
}


interface ThemeStore {
    isDark: boolean
    toggleTheme: () => void
    setTheme: (isDark: boolean) => void
}

