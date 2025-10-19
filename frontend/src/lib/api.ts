import axios from 'axios'
import type { Product, ChatRequest, ChatResponse, PaginatedResponse,  } from './types'

// API Base URL - change this when deploying
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
    'Content-Type': 'application/json',
    },
  timeout: 30000, // 30 seconds
})

// Add request interceptor for logging
api.interceptors.request.use(
    (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
    },
    (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
    }
)

// Add response interceptor for error handling
api.interceptors.response.use(
    (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
    },
    (error) => {
    console.error('API Response Error:', error.response?.data || error.message)
    return Promise.reject(error)
    }
)

/**
 * Products API
 */
export const productsApi = {
    /**
   * Get paginated products with filters
   */
    getProducts: async (params?: {
    skip?: number
    limit?: number
    category?: string
    brand?: string
    min_price?: number
    max_price?: number
    color?: string
    material?: string
    }): Promise<PaginatedResponse<Product>> => {
    const { data } = await api.get('/products', { params })
    return data
    },

/**
   * Get product by ID
   */
    getProduct: async (id: string): Promise<Product> => {
    const { data } = await api.get(`/products/${id}`)
    return data
    },

/**
   * Get all categories
   */
    getCategories: async (): Promise<{ total: number; categories: Record<string, number> }> => {
    const { data } = await api.get('/products/categories')
    return data
    },

/**
   * Get all brands
   */
    getBrands: async (): Promise<{ total: number; brands: Record<string, number> }> => {
    const { data } = await api.get('/products/brands')
    return data
    },

/**
   * Get product stats
   */
    getStats: async (): Promise<any> => {
    const { data } = await api.get('/products/stats/overview')
    return data
    },
}

/**
 * Chat API
 */
export const chatApi = {
/**
   * Send message to chatbot
   */
    sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const { data } = await api.post('/chat/message', request)
    return data
    },

/**
   * Get session history
   */
    getSession: async (sessionId: string): Promise<any> => {
    const { data } = await api.get(`/chat/session/${sessionId}`)
    return data
    },

/**
   * Clear session
   */
    clearSession: async (sessionId: string): Promise<any> => {
    const { data } = await api.delete(`/chat/session/${sessionId}`)
    return data
    },
}

/**
 * Health check
 */
export const healthCheck = async (): Promise<any> => {
    const { data } = await api.get('/health')
    return data
}

export default api