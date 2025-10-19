import { useState, useEffect } from 'react'
import { Search, SlidersHorizontal, X } from 'lucide-react'
import { ProductGrid } from '../../components/products/ProductGrid'
import { productsApi } from '../../lib/api'
import type { Product } from '../../lib/types'
import { debounce } from '../../lib/utils'

export function SearchPage() {
    const [products, setProducts] = useState<Product[]>([])
    const [loading, setLoading] = useState(true)
    const [searchQuery, setSearchQuery] = useState('')
    const [showFilters, setShowFilters] = useState(false)
    
  // Filters
    const [minPrice, setMinPrice] = useState<string>('')
    const [maxPrice, setMaxPrice] = useState<string>('')
    const [selectedCategory, setSelectedCategory] = useState<string>('')

  // Load products
    const loadProducts = async () => {
    setLoading(true)
    try {
        const params: any = { limit: 20 }
        
        if (minPrice) params.min_price = parseFloat(minPrice)
        if (maxPrice) params.max_price = parseFloat(maxPrice)
        if (selectedCategory) params.category = selectedCategory
        
        const response = await productsApi.getProducts(params)
        setProducts(response.items)
    } catch (error) {
        console.error('Failed to load products:', error)
    } finally {
        setLoading(false)
    }
    }

  // Load on mount
    useEffect(() => {
    loadProducts()
    }, [minPrice, maxPrice, selectedCategory])

  // Debounced search (future: integrate with text search)
    const handleSearch = debounce((query: string) => {
    // TODO: Integrate with text search endpoint
    console.log('Searching for:', query)
    }, 500)

    return (
    <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
        {/* Search Header */}
        <div className="mb-8">
            <h1 className="mb-2 text-3xl font-bold">Discover Furniture</h1>
            <p className="text-muted-foreground">
            Search through {products.length > 0 ? '305' : 'our'} products using AI-powered search
            </p>
        </div>

        {/* Search Bar */}
        <div className="mb-6 flex gap-3">
            <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
            <input
                type="text"
                placeholder="Search for furniture... (e.g., 'modern sofa under $500')"
                value={searchQuery}
                onChange={(e) => {
                setSearchQuery(e.target.value)
                handleSearch(e.target.value)
                }}
                className="h-14 w-full rounded-xl border border-border bg-background pl-12 pr-4 text-sm transition-all focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
            />
            {searchQuery && (
                <button
                onClick={() => setSearchQuery('')}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                <X className="h-5 w-5" />
                </button>
            )}
            </div>

          {/* Filter Toggle */}
            <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex h-14 items-center gap-2 rounded-xl border px-6 font-medium transition-all ${
                showFilters
                ? 'border-blue-500 bg-blue-500/10 text-blue-600 dark:text-blue-400'
                : 'border-border bg-background hover:bg-accent'
            }`}
            >
            <SlidersHorizontal className="h-5 w-5" />
            <span className="hidden sm:inline">Filters</span>
            </button>
        </div>

        {/* Filters Panel */}
        {showFilters && (
            <div className="mb-6 rounded-xl border border-border bg-card p-6 animate-in fade-in duration-200">
            <h3 className="mb-4 font-semibold">Filter Products</h3>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {/* Min Price */}
                <div>
                <label className="mb-2 block text-sm font-medium text-muted-foreground">
                    Min Price
                </label>
                <input
                    type="number"
                    placeholder="$0"
                    value={minPrice}
                    onChange={(e) => setMinPrice(e.target.value)}
                    className="h-10 w-full rounded-lg border border-border bg-background px-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                />
                </div>

              {/* Max Price */}
                <div>
                <label className="mb-2 block text-sm font-medium text-muted-foreground">
                    Max Price
                </label>
                <input
                    type="number"
                    placeholder="$1000"
                    value={maxPrice}
                    onChange={(e) => setMaxPrice(e.target.value)}
                    className="h-10 w-full rounded-lg border border-border bg-background px-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                />
                </div>

              {/* Category */}
                <div>
                <label className="mb-2 block text-sm font-medium text-muted-foreground">
                    Category
                </label>
                <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="h-10 w-full rounded-lg border border-border bg-background px-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                >
                    <option value="">All Categories</option>
                    <option value="sofa">Sofas</option>
                    <option value="chair">Chairs</option>
                    <option value="table">Tables</option>
                    <option value="bed">Beds</option>
                    <option value="desk">Desks</option>
                </select>
                </div>

              {/* Clear Filters */}
                <div className="flex items-end">
                <button
                    onClick={() => {
                    setMinPrice('')
                    setMaxPrice('')
                    setSelectedCategory('')
                    }}
                    className="h-10 w-full rounded-lg border border-border bg-background text-sm font-medium hover:bg-accent transition-colors"
                >
                    Clear Filters
                </button>
                </div>
            </div>
            </div>
        )}

        {/* Results Count */}
        <div className="mb-4 text-sm text-muted-foreground">
            {loading ? 'Loading...' : `Showing ${products.length} products`}
        </div>

        {/* Product Grid */}
        <ProductGrid
            products={products}
            loading={loading}
            onProductClick={(product) => {
            console.log('Clicked product:', product)
            // TODO: Open product detail modal/page
            }}
        />
        </div>
    </div>
    )
}