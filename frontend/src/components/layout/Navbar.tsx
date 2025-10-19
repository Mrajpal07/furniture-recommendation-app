import { Moon, Sun, Search, MessageSquare, Sparkles } from 'lucide-react'
import { useThemeStore } from '../../store/themeStore'
import { useChatStore } from '../../store/chatStore'

export function Navbar() {
    const { isDark, toggleTheme } = useThemeStore()
    const { openChat } = useChatStore()
    return (
    <nav className="sticky top-0 z-50 border-b border-border/40 bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo with gradient */}
            <div className="flex items-center gap-3">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 shadow-lg shadow-blue-500/30">
                <Sparkles className="h-5 w-5 text-white" />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
            </div>
            <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Furniture AI
                </h1>
                <p className="text-xs text-muted-foreground font-medium">Intelligent Discovery</p>
            </div>
            </div>

          {/* Navigation Links with hover effects */}
            <div className="hidden md:flex items-center gap-1">
            <NavLink icon={Search} active>Search</NavLink>
            <NavLink>Browse</NavLink>
            <NavLink>Compare</NavLink>
            <NavLink>Analytics</NavLink>
            </div>
            

            
          {/* Right side actions */}
            <div className="flex items-center gap-3">
            {/* AI Chat button with glow */}
            
            <button 
            onClick={openChat}
            className="group relative flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-blue-500/25 transition-all hover:shadow-xl hover:shadow-blue-500/40 hover:scale-[1.02] active:scale-[0.98]">
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-400 to-indigo-400 opacity-0 blur-xl transition-opacity group-hover:opacity-30" />
                <MessageSquare className="h-4 w-4 relative" />
                <span className="hidden sm:inline relative">AI Assistant</span>
            </button>


            {/* Dark mode toggle with animation */}
            <button
                onClick={toggleTheme}
                className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-border/50 bg-background/50 backdrop-blur-sm transition-all hover:border-border hover:bg-accent hover:scale-105 active:scale-95"
                aria-label="Toggle theme"
            >
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-yellow-400/10 to-orange-400/10 opacity-0 transition-opacity dark:from-blue-400/10 dark:to-indigo-400/10 hover:opacity-100" />
                {isDark ? (
                <Sun className="h-5 w-5 text-yellow-500 relative animate-in spin-in-180 duration-500" />
                ) : (
                <Moon className="h-5 w-5 text-slate-700 relative animate-in spin-in-180 duration-500" />
                )}
            </button>
            </div>
        </div>
        </div>
    </nav>
    )
}

// NavLink component with smooth hover effects
function NavLink({ 
    children, 
    icon: Icon, 
    active = false 
}: { 
    children: React.ReactNode
    icon?: any
    active?: boolean 
}) {
    return (
    <a
        href="#"
        className={`
            relative flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all
            ${active 
            ? 'text-foreground' 
            : 'text-muted-foreground hover:text-foreground'
        }
        `}
    >
        {Icon && <Icon className="h-4 w-4" />}
        {children}
        {active && (
            <div className="absolute inset-0 rounded-lg bg-accent/50 -z-10 animate-in fade-in duration-200" />
        )}
        <div className="absolute inset-0 rounded-lg bg-accent/0 transition-colors hover:bg-accent/30 -z-10" />
        </a>
    )
}