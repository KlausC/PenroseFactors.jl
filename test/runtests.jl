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

@testset "PenroseFactors" begin

    @testset "penrose accessors" for (n, m) in ((5, 5),), T in (Float64, ComplexF64)

        CC = testmatrix(m, n, T.([1.0, 0.1]))
        pf = penrose(CC; atol = 1e-13, rtol = 1e-15)
        U = pf.U
        R = pf.R
        V = pf.V
        p = pf.p
        P = pf.P

        @test CC[:,p] == CC * P
        @test U * R * V ≈ CC * P

        @test Matrix(pf) ≈ CC
    end


    @testset "penrose($m,$n) $T" for (n, m) in ((5, 3), (3, 5)), T in (Float64, ComplexF64)

        CC = testmatrix(m, n, T.([1.0; 0.5]))
        pf = penrose(CC)
        @test norm(pinv(pf) - pinv(CC)) < 1e-12
    end


end
