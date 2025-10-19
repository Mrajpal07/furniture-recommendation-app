import { Bot, User } from 'lucide-react'
import type { ChatMessage } from '../../lib/types'
import { ProductCard } from '../products/ProductCard'

interface MessageBubbleProps {
    message: ChatMessage
}

export function MessageBubble({ message }: MessageBubbleProps) {
    const isUser = message.role === 'user'
    
    return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
        <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
        isUser 
            ? 'bg-blue-600 text-white' 
            : 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white'
        }`}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </div>

      {/* Message Content */}
        <div className={`flex flex-col gap-2 max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Text Message */}
        <div className={`rounded-2xl px-4 py-2.5 ${
            isUser
            ? 'bg-blue-600 text-white'
            : 'bg-muted text-foreground border border-border'
        }`}>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Product Recommendations */}
        {message.products && message.products.length > 0 && (
            <div className="w-full space-y-2">
            <p className="text-xs font-medium text-muted-foreground px-2">
                Found {message.products.length} products:
            </p>
            <div className="grid grid-cols-2 gap-3">
                {message.products.slice(0, 4).map((product) => (
                <div key={product.uniq_id} className="scale-90 origin-top-left">
                    <ProductCard product={product} />
                </div>
                ))}
            </div>
            {message.products.length > 4 && (
                <p className="text-xs text-muted-foreground px-2">
                +{message.products.length - 4} more products
                </p>
            )}
            </div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-muted-foreground px-2">
            {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
        </div>
    </div>
    )
}