import type { ReactNode } from 'react'
import { Navbar } from './Navbar'
import { ChatPanel } from '../chat/ChatPanel'  // Add this import

interface LayoutProps {
    children: ReactNode
}

export function Layout({ children }: LayoutProps) {
    return (
    <div className="min-h-screen bg-background text-foreground">
    <Navbar />
    <main>{children}</main>
      <ChatPanel />  {/* Add this line */}
    </div>
    )
}