module PenroseFactors

using LinearAlgebra
export Penrose
export penrose, penrose!

using LinearAlgebra: AdjointQ, require_one_based_indexing, QRPackedQ, AbstractQ

import LinearAlgebra: *, \, pinv, rank, adjoint
import LinearAlgebra: rmul!, rdiv!, lmul!, ldiv!, qsize_check
import Base: size

struct Penrose{T,Q,V} <: Factorization{T}
    QR::Q
    σ::V
    Penrose(qrf::U, rqf::V) where {T,U<:Factorization{T},V} = new{T,U,V}(qrf, rqf)
end

size(pf::Penrose) = size(pf.QR)
size(pf::Penrose, i::Integer) = size(pf.QR, i)

function Base.getproperty(F::Penrose, s::Symbol)
    if s === :U
        qrf = getfield(F, :QR)
        r = size(getfield(F, :σ), 1)
        return QRPackedQ(view(qrf.Q.factors, :, 1:r), view(qrf.Q.τ, 1:r))
    elseif s === :R
        return getfield(F, :σ).R
    elseif s === :V
        getfield(F, :σ).Q
    else
        getfield(F, s)
    end
end

rtol(A::AbstractArray{T}, a) where T = min(size(A)...) * eps(real(float(T))) * iszero(a)

function penrose(A::AbstractMatrix; atol::Real=0, rtol::Real=rtol(A, atol))
    penrose!(copy(A); atol, rtol)
end

function penrose!(A::AbstractMatrix; atol::Real=0, rtol::Real=rtol(A, atol))
    m, n = size(A)
    qrf = qr!(A, ColumnNorm())
    r = rank(qrf; atol, rtol) # requires Julia1.12 copy impl. for lower versions
    qrf.τ[r+1:n] .= 0
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

LinearAlgebra.Matrix(pf::Penrose) = (pf.U*(pf.R*pf.V))[:, invperm(qrf.p)]

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

rank(pf::Penrose) = size(pf.σ, 1)

\(pf::Penrose, v::AbstractVector) = ldiv!(similar(v, size(pf, 2)), pf, copy(v))
\(pf::Penrose, v::AbstractMatrix) = ldiv!(similar(v, size(pf, 2), size(v, 2)), pf, copy(v))

function pinv(pf::Penrose{T}) where T
    m, n = size(pf)
    ldiv!(similar(pf.QR.factors, n, m), pf, Matrix{T}(I(m)))
end

"""
    rqr!(A::UpperTriangular, B::Matrix, τ::Vector)

Under the condition, that `A` is regular upper triangular of size `m`,
calculate a factorization `[A B] = R * Q` where `R` is a upper triangular matrix
 and `Q` is unitary.
After the calculation, `R` has overwritten the upper right part of `A`.
The `Q` is represented a the product `Q = prod H[i]` where
`H[i] = I - τ[i] * v[i]' * v[i]` and
`v[i,j] = `1 for i == j; 0 for j <= m; B[i,j] for j > m`.

Only the data fields of `A`, `B`, and `τ` are modified.
"""
function rq!(M::AbstractMatrix)
    require_one_based_indexing(M)
    m, n = size(M)
    rn = m+1:n
    τ = zeros(real(eltype(M)), m)
    rqr!(UpperTriangular(view(M, 1:m, 1:m)), view(M, 1:m, rn), τ)
    return QRPartial(M, τ)
end

function rqr!(A::UpperTriangular, B::AbstractMatrix{T}, τ::AbstractVector) where T
    require_one_based_indexing(A, B, τ)
    m, m1 = size(A)
    m == m1 || throw(DimensionMismatch(lazy"Matrix A must be quare but is [$m,$m1]"))
    m1, n = size(B)
    m == m1 || throw(DimensionMismatch(lazy"Matrix B must be [$m,:] but is [$m1,:]"))
    m1 = size(τ, 1)
    m == m1 || throw(DimensionMismatch(lazy"Matrix τ must be [$m] but is [$m1]"))
    B2 = axes(B, 2)

    if n == 0
        τ .= 0
        return
    end

    for k = m:-1:1
        β = sum(abs2, view(B, k, :))
        if iszero(β)
            τ[k] = 0
            continue
        end
        akk = A[k, k]
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
        B[k, :] .*= vi
        A[k, k] = -akk
        for j = k-1:-1:1
            bjk = A[j, k]'
            @inbounds for i in B2
                bjk += B[j, i]' * B[k, i]
            end
            bjk *= tauk
            A[j, k] -= bjk
            @inbounds for i in B2
                B[j, i] -= B[k, i] * bjk
            end
        end
    end
    return A, B, τ
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
    B, τ = Q.factors, Q.τ
    m, n = size(A)
    r, n1 = size(B)
    size_check(A, Q)
    rn = r+1:min(n, n1)
    rrange = updown(ud, r)
    for i = 1:m
        for k = rrange
            tauk = τ[k]
            iszero(tauk) && continue
            bik = A[i, k]
            @inbounds for ix in rn
                bik = B[k, ix]' * A[i, ix]
            end
            bik *= tauk
            A[i, k] -= bik
            @inbounds for ix in rn
                A[i, ix] -= B[k, ix] * bik
            end
        end
    end
    return A
end

function _lmul!(Q::QRPartialQ, A::AbstractVecOrMat, ud::Val)
    B, τ = Q.factors, Q.τ
    m, n = size(A, 1), size(A, 2)
    r, m1 = size(B)
    size_check(Q, A)
    rm = r+1:min(m, m1)
    rrange = updown(ud, r)
    for i = 1:n
        for k = rrange
            tauk = τ[k]
            iszero(tauk) && continue
            bik = A[k, i]
            @inbounds for ix in rm
                bik += B[k, ix]' * A[ix, i]
            end
            bik *= tauk
            A[k, i] -= bik
            @inbounds for ix in rm
                A[ix, i] -= B[k, ix] * bik
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

function testmatrix(m::Int, n::Int, d::AbstractVector)
    r = length(d)
    U = qr!(rand(m, r)).Q
    V = qr!(rand(n, r)).Q
    U * Diagonal(d) * V'
end

end # PenroseFactors
