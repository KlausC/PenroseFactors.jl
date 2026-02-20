module Fractals
using StaticArrays

export LinearFractal, genfractal


struct LinearFractal{N,T,V<:AbstractVector{T}}
    x::V
    y::V
    h::V
    c::V
    s::V

    function LinearFractal(x, y)
        N = length(x)
        N == length(y) || throw(ArgumentError("x and y coordinates require same length"))
        T = promote_type(eltype(x), eltype(y))
        V = SVector{N,T}
        x = convert(V, x)
        y = convert(V, y)
        h = convert(V, [hypot(x[mod1(k + 1, N)] - x[k], y[mod1(k + 1, N)] - y[k]) for k = 1:N])
        c = convert(V, [(x[mod1(k + 1, N)] - x[k]) / h[k] for k = 1:N])
        s = convert(V, [(y[mod1(k + 1, N)] - y[k]) / h[k] for k = 1:N])
        new{N - 1,T,V}(x, y, h, c, s)
    end
end

xx(p::LinearFractal, k) = p.x[k+1]
yy(p::LinearFractal, k) = p.y[k+1]
hh(p::LinearFractal, k) = p.h[k+1]
cc(p::LinearFractal, k) = p.c[k+1]
ss(p::LinearFractal, k) = p.s[k+1]

function genfractal(p::LinearFractal{N,T}, n::Integer, tt::AbstractFloat) where {N,T}
    n >= 0 || throw(ArgumentError("n must not be negative"))
    N > 0 || throw(ArgumentError("N must be positive"))
    n = min(n, Integer(ceil(log(N, inv(eps(eltype(tt)))))))
    H = hh(p, N)
    t1, tt = divrem(tt, one(T))
    t = intprod(tt, N, n)
    x, y = zeros(T, 2)
    for _ = 1:n
        t, k = divrem(t, N)
        v, w, h, c, s = xx(p, k), yy(p, k), hh(p, k) / H, cc(p, k), ss(p, k)
        x, y = (c * x - s * y) * h, (s * x + c * y) * h
        x, y = x + v, y + w
    end
    if t1 > 0
        x += xx(p, N) * t1
        y += yy(p, N) * t1
    end
    x, y
end

# calculate Integer(floor(t * N^n)) a bit more accurate
function intprod(t::AbstractFloat, N::Integer, n::Integer)
    if n <= 8 && N <= 90
        Integer(floor(t * N^n))
    else
        n1 = n ÷ 2
        Nn1 = N^n1
        Nn2 = N^(n - n1)
        t1 = Integer(floor(t * Nn1))
        t2 = t * Nn1 - t1
        Integer(floor(t2 * Nn2)) + t1 * Nn2
    end
end

end
