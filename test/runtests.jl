using PenroseFactors
using Test
using LinearAlgebra

testmatrix(m::Int, n::Int, r::Int) = testmatrix(m, n, [2.0^(1 - j) for j = 1:r])

function testmatrix(m::Int, n::Int, d::AbstractVector{T}) where T<:Number
    r = length(d)
    U = qr!(rand(T, m, r)).Q
    V = qr!(rand(T, n, r)).Q
    U * Diagonal(d) * V'
end

ALPHABETA = (
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-0.3, 0.4),
    (0.4 + 0.3im, 0.1),
    (-0.4 + 0.3im, 0.1),
    (0.4 - 0.3im, 0.1),
    (-0.4 - 0.3im, 0.1),
    (0.3 + 0.4im, 0.1),
    (-0.3 + 0.4im, 0.1),
    (0.3 - 0.4im, 0.1),
    (-0.3 - 0.4im, 0.1),
)

using PenroseFactors: tau_v_from_a_b
testtau(tau, v, b) = 2real(tau) * abs2(v) - abs2(tau) * (abs2(v) + abs2(b))
testvau(tau, v, a, b) = abs2(v) - (a * v' + abs2(b)) * tau
testnorm(t, v, a, b) = abs(a - v) - hypot(a, b)

@testset "PenroseFactors" begin

    @testset "tau_and_v($a, $b, $kind)" for (a, b) in ALPHABETA, kind in 1:3
        t, v = tau_v_from_a_b(a, b, Val(kind))
        @test abs(testtau(t, v, b)) < 1e-15
        @test abs(testvau(t, v, a, b)) < 1e-15
        @test abs(testnorm(t, v, a, b)) < 1e-15
        b != 0 && @test abs(v) >= hypot(a, b)
    end

    @testset "penrose accessors" for (n, m) in ((5, 5),), T in (Float64, ComplexF64)

        CC = testmatrix(m, n, T.([1.0, 0.1]))
        pf = penrose(CC; atol=1e-13, rtol=1e-15)
        U = pf.U
        R = pf.R
        V = pf.V
        p = pf.p
        P = pf.P

        @test CC[:, p] == CC * P
        @test U * R * V ≈ CC * P

        @test Matrix(pf) ≈ CC
    end


    @testset "penrose($m,$n) $T" for (n, m) in ((5, 3), (3, 5)), T in (Float64, ComplexF64)

        CC = testmatrix(m, n, T.([1.0; 0.5]))
        pf = penrose(CC)
        @test norm(pinv(pf) - pinv(CC)) < 1e-12
    end


end
