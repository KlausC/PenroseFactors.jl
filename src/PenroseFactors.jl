module PenroseFactors

using LinearAlgebra
export Penrose
export penrose, penrose!
export householder_column, householder_standard!, householder_row, householder_bidiag, MGS!

using LinearAlgebra: AdjointQ, require_one_based_indexing, QRPackedQ, AbstractQ

import LinearAlgebra: *, \, pinv, rank, adjoint
import LinearAlgebra: rmul!, rdiv!, lmul!, ldiv!, qsize_check
import Base: size

include("householder.jl")

struct Penrose{T,Q,V} <: Factorization{T}
    QR::Q
    RQ::V
    Penrose(qrf::U, rqf::V) where {T,U<:Factorization{T},V} = new{T,U,V}(qrf, rqf)
end

size(pf::Penrose) = size(pf.QR)
size(pf::Penrose, i::Integer) = size(pf.QR, i)

function Base.getproperty(F::Penrose{T}, s::Symbol) where T
    if s === :U
        Q = getfield(F, :QR).Q
        r = size(getfield(F, :RQ), 1)
        r == length(Q.τ) ? Q : QRPackedQ(view(Q.factors, :, 1:r), view(Q.τ, 1:r))
    elseif s === :R
        return getfield(F, :RQ).R
    elseif s === :V
        getfield(F, :RQ).Q
    elseif s == :p
        getfield(F, :QR).p
    elseif s === :P
        p = F.p
        n = length(p)
        P = zeros(T, n, n)
        for i in 1:n
            P[p[i], i] = one(T)
        end
        return P
    else
        getfield(F, s)
    end
end

rtol(A::AbstractArray{T}, a) where T = min(size(A)...) * eps(real(float(T))) * iszero(a)

function penrose(A::AbstractMatrix; atol::Real=0, rtol::Real=rtol(A, atol), rk::Int=-1)
    penrose!(copy(A); atol, rtol, rk)
end

function penrose!(A::AbstractMatrix; atol::Real=0, rtol::Real=rtol(A, atol), rk::Int=-1)
    m, n = size(A)
    qrf = qr!(A, ColumnNorm())
    r = rk >= 0 ? rk : _rank(qrf; atol, rtol) # requires Julia1.12 copy impl. for lower versions
    qrf.τ[r+1:min(n, m)] .= 0
    #qrf.factors[r+1:m, r+1:n] .= 0
    lqf = rq!(view(A, 1:r, :))
    return Penrose(qrf, lqf)
end

struct QRPartial{T,U,V} <: Factorization{T}
    factors::U
    τ::V
    QRPartial(f::U, t::V) where {T,U<:AbstractMatrix{T},V<:AbstractVector{<:Real}} =
        new{T,U,V}(f, t)
end

struct QRPartialQ{T,U,V} <: LinearAlgebra.AbstractQ{T}
    factors::U
    τ::V
    QRPartialQ(f::U, t::V) where {T,U<:AbstractMatrix{T},V<:AbstractVector{<:Real}} =
        new{T,U,V}(f, t)
end

function Base.getproperty(F::QRPartial, s::Symbol)
    if s === :Q
        return QRPartialQ(getfield(F, :factors), getfield(F, :τ))
    elseif s === :R
        m = axes(F.factors, 1)
        return triu!(getfield(F, :factors)[m, m])
    else
        getfield(F, s)
    end
end

Base.size(F::QRPartial, dims...) = size(F.factors, dims...)
Base.size(Q::QRPartialQ) = (n = size(Q.factors, 2); (n, n))

LinearAlgebra.Matrix(pf::Penrose) = (pf.U*(pf.R*pf.V))[:, invperm(pf.p)]

LinearAlgebra.ldiv!(w::AbstractVector, pf::Penrose, v::AbstractVector) = _ldiv!(w, pf, v)
LinearAlgebra.ldiv!(w::AbstractMatrix, pf::Penrose, v::AbstractMatrix) = _ldiv!(w, pf, v)

function _ldiv!(w, pf::Penrose, v)
    m, n = size(pf)
    m == size(v, 1) || throw(DimensionMismatch(lazy""))
    n == size(w, 1) || throw(DimensionMismatch(lazy""))
    size(v, 2) == size(w, 2) || throw(DimensionMismatch(lazy""))

    r = rank(pf)
    p = pf.QR.p
    w .= 0
    W = view(w, p, :)
    ldiv!(pf.U, v)
    ldiv!(UpperTriangular(pf.R), view(v, 1:r, :))
    W[1:r, :] .= view(v, 1:r, :)
    ldiv!(pf.V, W)
    return w # view(w, invperm(p), :)
end

rank(pf::Penrose) = size(pf.RQ, 1)

\(pf::Penrose, v::AbstractVector) = ldiv!(similar(v, size(pf, 2)), pf, copy(v))
\(pf::Penrose, v::AbstractMatrix) = ldiv!(similar(v, size(pf, 2), size(v, 2)), pf, copy(v))

function pinv(pf::Penrose{T}) where T
    m, n = size(pf)
    ldiv!(similar(pf.QR.factors, n, m), pf, Matrix{T}(I(m)))
end

"""
    rq!(M::AbstractMatrix)::QRPartial

Perform a R-Q factorization of flat matrix `M` in-memory.

Actually only upper part of `M` is used or modified.

`M == [A B]; m, n = size(M)`

A real vector `τ` of length `size(M, 1)` is allocated

Under the condition, that `A` is regular upper triangular of size `m`,
calculate a factorization `[A B] = R * Q` where `R` is a upper triangular matrix
and `Q` is unitary.

After the calculation, `R` has overwritten the upper right part of `A` in `M`.
The `Q` is represented a the product `Q = prod H[i]` where
`H[i] = I - τ[i] * v[i]' * v[i]` and
`v[i,j] = `1 for i == j; 0 for j <= m; M[i,j] for j > m`.

An `AbstractQ` object is returned, which contains all relevent information of `M` and `τ`.
"""
function rq!(M::AbstractMatrix{T}) where T
    require_one_based_indexing(M)
    m, m1 = size(M)
    m <= m1 || throw(DimensionMismatch(lazy"Matrix A must be flat but is [$m,$m1]"))
    B2 = m+1:m1
    τ = zeros(real(T), m)
    if m == m1
        return QRPartial(M, τ)
    end

    for k = m:-1:1
        β = sum(abs2, view(M, k, B2))
        if iszero(β)
            τ[k] = 0
            continue
        end
        akk = M[k, k]
        α = abs2(akk)
        if iszero(α)
            v = akk = sqrt(β)
            tauk = one(v)
        else
            ρ = sqrt(β / α + one(α))
            v = ρ * akk
            akk, v = v, v + akk
            tauk = 1 / ρ + 1
        end
        vi = inv(v)
        τ[k] = tauk
        for ix in B2
            M[k, ix] *= vi
        end
        M[k, k] = -akk
        @inbounds for j = k-1:-1:1
            bjk = M[j, k]
            for i in B2
                bjk += M[j, i] * M[k, i]'
            end
            bjk *= tauk
            M[j, k] -= bjk
            for i in B2
                M[j, i] -= M[k, i] * bjk
            end
        end
    end
    return QRPartial(M, τ)
end

rmul!(A::AbstractMatrix, Q::QRPartialQ) = _rmul!(A, Q, Val(:UP))
rmul!(A::AbstractMatrix, Q::AdjointQ{<:Any,<:QRPartialQ}) = _rmul!(A, Q.Q, Val(:DOWN))
rdiv!(A::AbstractMatrix, Q::QRPartialQ) = _rmul!(A, Q, Val(:DOWN))
rdiv!(A::AbstractMatrix, Q::AdjointQ{<:Any,<:QRPartialQ}) = _rmul!(A, Q.Q, Val(:UP))

lmul!(Q::QRPartialQ, A::AbstractVecOrMat) = _lmul!(Q, A, Val(:DOWN))
lmul!(Q::AdjointQ{<:Any,<:QRPartialQ}, A::AbstractVecOrMat) = _lmul!(Q.Q, A, Val(:UP))
#ldiv!(Q::QRPartialQ, A::AbstractMatrix) = _lmul!(Q, A, Val(:UP))
#ldiv!(Q::AdjointQ{<:Any,<:QRPartialQ}, A::AbstractVecOrMat) = _lmul!(Q.Q, A, Val(:DOWN))

function _rmul!(A::AbstractMatrix, Q::QRPartialQ, ud::Val)
    require_one_based_indexing(A)
    B, τ = Q.factors, Q.τ
    m, n = size(A)
    r, n1 = size(B)
    size_check(A, Q)
    rn = r+1:min(n, n1)
    rrange = updown(ud, r)
    @inbounds for i = 1:m
        for k = rrange
            tauk = τ[k]
            iszero(tauk) && continue
            bik = A[i, k]
            for ix in rn
                bik += B[k, ix]' * A[i, ix]
            end
            bik *= tauk
            A[i, k] -= bik
            for ix in rn
                A[i, ix] -= B[k, ix] * bik
            end
        end
    end
    return A
end

function _lmul!(Q::QRPartialQ, A::AbstractVecOrMat, ud::Val)
    require_one_based_indexing(A)
    B, τ = Q.factors, Q.τ
    m, n = size(A, 1), size(A, 2)
    r, m1 = size(B)
    size_check(Q, A)
    rm = r+1:min(m, m1)
    rrange = updown(ud, r)
    @inbounds for i = 1:n
        for k = rrange
            tauk = τ[k]
            iszero(tauk) && continue
            bik = A[k, i]
            for ix in rm
                bik += B[k, ix] * A[ix, i]
            end
            bik *= tauk
            A[k, i] -= bik
            for ix in rm
                A[ix, i] -= B[k, ix]' * bik
            end
        end
    end
    return A
end

updown(::Val{:UP}, r) = 1:r
updown(::Val{:DOWN}, r) = r:-1:1

# flexible left-multiplication (and adjoint right-multiplication)
qsize_check(adjQ::Adjoint{<:QRPartialQ}, B::AbstractVecOrMat) =
    (Q = adjQ.Q;
    size(B, 1) in size(Q.factors) ||
        throw(DimensionMismatch(lazy"first dimension of B, $(size(B,1)), must equal one of the dimensions of Q, $(size(Q.factors))")))
qsize_check(A::AbstractMatrix, Q::QRPartialQ) =
    (size(A, 2) in size(Q.factors) ||
     throw(DimensionMismatch(lazy"second dimension of A, $(size(A,2)), must equal one of the dimensions of Q, $(size(Q.factors))")))

function size_check(A::Union{AbstractArray,AbstractQ}, B::Union{AbstractArray,AbstractQ})
    mA = size(A, 2)
    mB = size(B, 1)
    if mA != mB
        throw(DimensionMismatch(lazy"matrix A has dimensions (:,$mA) but B has dimensions ($mB,:)"))
    end
end


testmatrix(m::Int, n::Int, r::Int) = testmatrix(m, n, [2.0^(1 - j) for j = 1:r])

function testmatrix(m::Int, n::Int, d::AbstractVector{T}) where T<:Number
    r = length(d)
    U = qr!(rand(T, m, r)).Q
    V = qr!(rand(T, n, r)).Q
    U * Diagonal(d) * V'
end

if VERSION >= v"1.12"
    _rank(A::QRPivoted; atol::Real=0, rtol::Real=rtol(A, atol)) = rank(A; atol, rtol)
else
    """
        rank(A::QRPivoted{<:Any, T}; atol::Real=0, rtol::Real=min(n,m)*ϵ) where {T}

    Compute the numerical rank of the QR factorization `A` by counting how many diagonal entries of
    `A.factors` are greater than `max(atol, rtol*Δ₁)` where `Δ₁` is the largest calculated such entry.
    This is similar to the [`rank(::AbstractMatrix)`](@ref) method insofar as it counts the number of
    (numerically) nonzero coefficients from a matrix factorization, although the default method uses an
    SVD instead of a QR factorization. Like [`rank(::SVD)`](@ref), this method also re-uses an existing
    matrix factorization.

    Using a QR factorization to compute rank should typically produce the same result as using SVD,
    although it may be more prone to overestimating the rank in pathological cases where the matrix is
    ill-conditioned. It is also worth noting that it is generally faster to compute a QR factorization
    than it is to compute an SVD, so this method may be preferred when performance is a concern.

    `atol` and `rtol` are the absolute and relative tolerances, respectively.
    The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `A`
    and `ϵ` is the [`eps`](@ref) of the element type of `A`.

    !!! compat "Julia 1.12"
        The `rank(::QRPivoted)` method requires at least Julia 1.12.
    """
    function _rank(A::QRPivoted; atol::Real=0, rtol::Real=rtol(A, atol))
        m = min(size(A)...)
        m == 0 && return 0
        tol = max(atol, rtol * abs(A.factors[1, 1]))
        return something(findfirst(i -> abs(A.factors[i, i]) <= tol, 1:m), m + 1) - 1
    end
end

end # PenroseFactors
