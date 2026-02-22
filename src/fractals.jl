module Fractals
using StaticArrays

export LinearFractal, genfractal


struct LinearFractal{N,T,V<:AbstractVector{T}}
    x::V
    y::V
    h::V
    c::V
    s::V
    yfactor::V

    function LinearFractal(x, y, yf=ones(length(x)))
        N = length(x)
        N == length(y) || throw(ArgumentError("x and y coordinates require same length"))
        T = float(promote_type(eltype(x), eltype(y)))
        V = SVector{N,T}
        x = convert(V, x)
        y = convert(V, y)
        h = convert(V, [hypot(x[mod1(k + 1, N)] - x[k], y[mod1(k + 1, N)] - y[k]) for k = 1:N])
        c = convert(V, [(x[mod1(k + 1, N)] - x[k]) / h[k] for k = 1:N])
        s = convert(V, [(y[mod1(k + 1, N)] - y[k]) / h[k] for k = 1:N])
        yf = convert(V, yf)
        new{N - 1,T,V}(x, y, h, c, s, yf)
    end
end

Base.show(io::IO, ::LinearFractal{N,T}) where {N,T} = print(io, "LinearFractal{$N,$T}")

xx(p::LinearFractal, k) = p.x[k+1]
yy(p::LinearFractal, k) = p.y[k+1]
hh(p::LinearFractal, k) = p.h[k+1]
cc(p::LinearFractal, k) = p.c[k+1]
ss(p::LinearFractal, k) = p.s[k+1]
yf(p::LinearFractal, k) = p.yfactor[k+1]

function genfractal(p::LinearFractal{N,T}, n::Integer, tt::Real) where {N,T}
    n >= 0 || throw(ArgumentError("n must not be negative"))
    N > 0 || throw(ArgumentError("N must be positive"))
    n = min(n, Integer(ceil(log(N, inv(eps(float(eltype(tt)))))))) # N^n not larger that precision
    H = hh(p, N)
    t1, tt = fldmod(tt, one(T))
    t, d = intprod(tt, N, n)
    x, y = xx(p, N) * d, yy(p, N) * d
    for _ = 1:n
        t, k = divrem(t, N)
        v, w, h, c, s, yp = xx(p, k), yy(p, k), hh(p, k) / H, cc(p, k), ss(p, k), yf(p, k)
        y = y * yp
        x, y = (c * x - s * y) * h, (s * x + c * y) * h
        x, y = x + v, y + w
    end
    if !iszero(t1)
        x += xx(p, N) * t1
        y += yy(p, N) * t1
    end
    x, y
end

# calculate Integer(floor(t * N^n)) a bit more accurate
function intprod(t::AbstractFloat, N::Integer, n::Integer)
    if n <= 8 && N <= 90
        t, d = fldmod(t * N^n, one(t))
        Integer(t), d
    else
        n1 = n ÷ 2
        Nn1 = N^n1
        Nn2 = N^(n - n1)
        t1, d1 = fldmod(t * Nn1, one(t))
        t2, d = fldmod(d1 * Nn2, one(t))
        Integer(t1) * Nn2 + Integer(t2), d
    end
end

end
