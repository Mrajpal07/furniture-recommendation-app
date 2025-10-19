import { ProductCard } from './ProductCard'
import type { Product } from '../../lib/types'


interface ProductGridProps {
    products: Product[]
    loading?: boolean
    onProductClick?: (product: Product) => void
}

export function ProductGrid({ products, loading, onProductClick }: ProductGridProps) {
    if (loading) {
    return (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
            <ProductCardSkeleton key={i} />
        ))}
        </div>
    )
    }

    if (products.length === 0) {
    return (
        <div className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border/50 bg-muted/30 py-20">
        <div className="mb-4 text-6xl">🔍</div>
        <h3 className="mb-2 text-xl font-semibold">No products found</h3>
        <p className="text-sm text-muted-foreground">Try adjusting your filters or search query</p>
        </div>
    )
    }

    return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {products.map((product) => (
        <ProductCard
            key={product.uniq_id}
            product={product}
            onClick={() => onProductClick?.(product)}
        />
        ) )}
    </div>
    )
}

// Loading skeleton
function ProductCardSkeleton() {
    return (
    <div className="overflow-hidden rounded-2xl border border-border/50 bg-card">
    <div className="aspect-square animate-pulse bg-muted/50" />
    <div className="p-4 space-y-3">
        <div className="h-3 w-20 animate-pulse rounded bg-muted" />
        <div className="h-4 w-full animate-pulse rounded bg-muted" />
        <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
        <div className="flex justify-between">
        <div className="h-6 w-24 animate-pulse rounded bg-muted" />
        <div className="h-6 w-16 animate-pulse rounded bg-muted" />
        </div>
    </div>
    </div>
    )
}