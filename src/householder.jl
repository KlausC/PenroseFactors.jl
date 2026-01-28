

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

function householder_standard!(A, K=Val(1))
    m, n = size(A)
    τ = similar(A, min(m, n))
    for col = 1:min(m, n)
        τ[col] = householder_column(A, col, col, col+1:m, col+1:n)
    end
    return τ, A
end
