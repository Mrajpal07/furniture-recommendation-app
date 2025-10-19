/**
 * Product from API (matches backend structure)
 */
export interface Product {
    uniq_id: string
    title: string
    brand: string
    price?: string | number  // Can be string from backend
    price_numeric?: number    // Actual numeric price
    categories: string
    category?: string
    description: string
    images?: string           // Backend returns this
    image_url?: string        // We'll map to this
    score?: number
    match_type?: string
    color?: string
    material?: string
    manufacturer?: string
}

/**
 * Search filters
 */
export interface SearchFilters {
    min_price?: number
    max_price?: number
    categories?: string[]
    brands?: string[]
    colors?: string[]
    materials?: string[]
}

/**
 * Chat message
 */
export interface ChatMessage {
    role: 'user' | 'assistant'
    content: string
    timestamp?: string
    products?: Product[]
}

/**
 * Chat request/response
 */
export interface ChatRequest {
    message: string
    session_id: string
}

export interface ChatResponse {
    message: string
    session_id: string
    intent: string
    filters?: SearchFilters
    products?: Product[]
    }

/**
 * API response wrappers
 */
export interface PaginatedResponse<T> {
    total: number
    skip: number
    limit: number
    items: T[]
    has_more: boolean
}