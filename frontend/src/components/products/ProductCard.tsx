import type { Product } from '../../lib/types'

interface ProductCardProps {
    product: Product
    onClick?: () => void
    }

export function ProductCard({ product, onClick }: ProductCardProps) {
  // Format price - handle both price and price_numeric
    const formatPrice = (product: Product) => {
    let priceValue = 0
    
    // Try price_numeric first (backend uses this)
    if (product.price_numeric && product.price_numeric > 0) {
        priceValue = product.price_numeric
    } 
    // Try parsing price string
    else if (product.price) {
        const priceStr = String(product.price)
        const cleaned = priceStr.replace(/[^0-9.]/g, '')
        priceValue = parseFloat(cleaned) || 0
    }
    
    if (priceValue === 0) {
        return 'Price N/A'
    }
    
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
    }).format(priceValue)
    }

  // Get image URL - backend uses 'images' field
    const getImageUrl = (product: Product) => {
    // Try image_url first
    if (product.image_url) {
        return product.image_url
    }
    
    // Try images field (backend format)
    if (product.images) {
        const imagesStr = String(product.images)
      // Extract first URL from string like "['url1', 'url2']"
        const match = imagesStr.match(/https?:\/\/[^\s,\]'"]+/)
        if (match) {
        return match[0]
        }
    }
    
    return null
    }

    const imageUrl = getImageUrl(product)
    const price = formatPrice(product)

    return (
    <div 
        onClick={onClick}
        className="group relative overflow-hidden rounded-2xl border border-border/50 bg-card backdrop-blur-sm transition-all hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 cursor-pointer"
        >
      {/* Image */}
        <div className="relative aspect-square overflow-hidden bg-muted/30">
        {imageUrl ? (
            <img
            src={imageUrl}
            alt={product.title}
            className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-110"
            onError={(e) => {
                const target = e.target as HTMLImageElement
                target.style.display = 'none'
                const parent = target.parentElement
                if (parent) {
                parent.innerHTML = '<div class="flex h-full w-full items-center justify-center bg-gradient-to-br from-muted to-muted/50"><span class="text-6xl">🛋️</span></div>'
                }
            }}
            />
        ) : (
            <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                <span className="text-6xl">🛋️</span>
            </div>
        )}
        
        {/* AI Score Badge */}
        {product.score && product.score > 0.8 && (
            <div className="absolute left-2 top-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 px-2.5 py-1 text-xs font-semibold text-white shadow-lg">
                {Math.round(product.score * 100)}% Match
            </div>
        )}
        </div>

      {/* Content */}
        <div className="p-4">
        {/* Brand */}
        <div className="mb-1 text-xs font-medium text-muted-foreground uppercase tracking-wide">
            {product.brand || 'Unknown Brand'}
        </div>

        {/* Title */}
        <h3 className="mb-2 line-clamp-2 text-sm font-semibold text-foreground group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors min-h-[2.5rem]">
            {product.title}
        </h3>

        {/* Category Tags */}
        {(product.categories || product.category) && (
            <div className="mb-3 flex flex-wrap gap-1">
            {(product.categories || product.category || '').split(',').slice(0, 2).map((cat, i) => (
                cat.trim() && (
                <span
                    key={i}
                    className="rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground"
                >
                    {cat.trim()}
                </span>
                )
            ))}
            </div>
        )}

        {/* Price */}
        <div className="flex items-center justify-between">
            <div className="text-xl font-bold bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent">
            {price}
            </div>
            {product.match_type && (
            <div className="rounded-full bg-blue-500/10 px-2 py-1 text-xs font-medium text-blue-600 dark:text-blue-400">
                {product.match_type === 'text' ? '📝' : product.match_type === 'image' ? '🖼️' : '🎯'}
            </div>
            )}
        </div>
        </div>

      {/* Hover gradient effect */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-blue-500/0 via-indigo-500/0 to-purple-500/0 opacity-0 transition-opacity group-hover:opacity-20 pointer-events-none" />
    </div>
    )
}