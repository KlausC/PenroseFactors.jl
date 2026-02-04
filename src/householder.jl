

"""
    tau_v_from_a_b(a::Complex, b::Real, Val(N)) where N in {1,2,3}

    - In principle v max by any complex number with abs(v - a) = hypot(a, b)
    -  Of special interest
    - 1. abs(v) is maximal, τ is real - Bulirsch-Stoer
    - 2. v - a is real - used in LAPACK qr
    - 3. v is real or pure imaginary - less multiplication effort with vectors
"""
function tau_v_from_a_b(a, b, ::Val) end

function tau_v_from_a_b(a::T, b::Real, ::Val{1}) where T<:Union{Real,Complex}
    if iszero(b)
        return zero(real(T)), b
    end
    if iszero(a)
        return oneunit(real(T)), b
    end
    v = (a isa Real ? copysign(hypot(a, b), a) : hypot(a, b) * sign(a)) + a
    τ = 2 / ((b / abs(v))^2 + 1)
    return τ, v
end

function tau_v_from_a_b(a::T, b::Real, ::Val{2}) where T<:Union{Real,Complex}
    if iszero(b)
        return zero(real(T)), b
    end
    v = copysign(hypot(a, b), real(a)) + a
    τ = inv(((b / abs(v))^2 + 1) / 2 + imag(a / v) * im)
    return τ, v
end

function tau_v_from_a_b(a::T, b::Real, ::Val{3}) where T<:Union{Real,Complex}
    if iszero(b)
        return zero(real(T)), b
    end
    if T <: Real
        ax = a
        s = 1
    else
        ar, ai = real(a), imag(a)
        if iszero(ai) || abs(ar) >= abs(ai)
            ax = ar
            s = 1
        else
            ax = ai
            s = im
        end
    end
    v = (copysign(hypot(ax, b), ax) + ax) * s
    τ = inv(((b / abs(v))^2 + 1) / 2 + imag(a / v) * im)
    return τ, v
end

function householder_column(A::AbstractArray, col::Integer, i::Integer, rr, cr, K=Val(1))
    a = A[i, col]
    B = view(A, rr, col)
    b = norm(B)

    τ, v = tau_v_from_a_b(a, b, K)
    iszero(τ) && return τ
    A[i, col] -= v
    vinv = inv(v)
    for k in rr
        A[k, col] *= vinv
    end
    for j in cr
        d = (dot(B, view(A, rr, j)) + A[i, j]) * τ
        A[i, j] -= d
        for k in rr
            A[k, j] -= A[k, col] * d
        end
    end
    return τ
end
function householder_row(A::AbstractArray, row::Integer, i::Integer, cc, rc, K=Val(1))
    a = A[row, i]
    B = view(A, row, cc)
    b = norm(B)

    τ, v = tau_v_from_a_b(a, b, K)
    iszero(τ) && return τ
    A[row, i] -= v
    vinv = inv(v)
    for k in cc
        A[row, k] *= vinv
    end
    for j in rc
        d = (dot(B, view(A, j, cc)) + A[j, i]) * τ
        A[j, i] -= d
        for k in cc
            A[j, k] -= A[row, k] * d
        end
    end
    return τ
end

function householder_standard!(A, K=Val(1))
    m, n = size(A)
    τ = similar(A, min(m, n))
    for col = 1:min(m, n)
        τ[col] = householder_column(A, col, col, col+1:m, col+1:n, K)
    end
    return τ, A
end

householder_bidiag(A, K=Val(1)) = householder_bidiag!(copy(A), K)

function householder_bidiag!(A, K=Val(1))
    m, n = size(A)
    l = min(n, m + 1)
    d = similar(A, l)
    u = similar(A, l - 1)
    for i = 1:min(n, m)
        householder_column(A, i, i, i+1:m, i+1:n, K)
        d[i] = A[i, i]
        if i < n
            householder_row(A, i, i + 1, i+2:n, i+1:m, K)
            u[i] = A[i, i+1]
        end
    end
    d[l] = l <= min(n, m) ? A[l, l] : zero(eltype(A))
    return Bidiagonal(d, u, :U)
end


"""
    MGS!(A::Matrix) -> V, S, d

Modified Gram-Schmidt applied to input matirx `A`.

Returns orthogonalized `V`, upper unit triangular `S`, and `d` diag of `V'V`.

A factorization `A = V * S` takes place.

A orthonormal factorization `A = Q * R` derives from that with
`Q = V ./ Diagonal(sqrt.(d)` and `R = Diagonal(sqrt.(d)) * S`
"""
function MGS!(A)
    m, n = size(A)
    R = zeros(eltype(A), m, n)
    d = zeros(eltype(A), n)
    for j = 1:n
        j <= m && (R[j, j] = 1)
        for i = 1:min(j - 1, m)
            rr = dot(view(A, :, j), view(A, :, i)) / d[i]
            R[i, j] = rr
            for k = 1:m
                A[k, j] -= A[k, i] * rr
            end
        end
        d[j] = sum(abs2, view(A, :, j))
    end
    return A, R, d
end

function MGS!(A, ::ColumnNorm)
    m, n = size(A)
    mn = min(n, m)
    p = [i for i in 1:n]
    R = zeros(eltype(A), mn, n)
    d = zeros(eltype(A), n)
    for j = 1:n
        d[j] = sum(abs2, view(A, :, j))
    end
    for i = 1:mn
        di, piv = findmax(view(d, i:n))
        piv += i - 1
        if piv != i
            switchcol!(A, 1:m, i, piv)
            switchcol!(R, 1:i-1, i, piv)
            d[i], d[piv] = d[piv], d[i]
            p[i], p[piv] = p[piv], p[i]
        end
        R[i,i] = 1
        for j = i+1:n
            aa = dot(view(A, :, j), view(A, :, i))
            rr = aa / di
            R[i, j] = rr
            for k = 1:m
                A[k, j] -= A[k, i] * rr
            end
            d[j] = sum(abs2, view(A, :, j))
            #d[j] -= conj(aa) * rr # inexact
        end
    end
    return A, R, d, p
end

function switchcol!(A, range, i, j)
    for k in range
        A[k,i], A[k,j] = A[k,j], A[k,i]
    end
end
