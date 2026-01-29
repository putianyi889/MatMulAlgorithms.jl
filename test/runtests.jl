using MatMulAlgorithms
using Random
using Test

@testset "MatMulAlgorithms.jl" begin
    A = rand(800,1000)
    B = rand(1000,1200)
    C = rand(800,1200)

    Y = similar(C)

    @testset "stevengj" begin
        copyto!(Y,C)
        @test stevengj_mul!(Y, A, B) ≈ C+A*B
    end

    @testset "winograd" begin
        copyto!(Y,C)
        @test isapprox(winograd_mul!(Y, A, B), C+A*B, rtol=1e-2)
    end

    @testset "octavian" begin
        @test octavian_mul!(Y, A, B) ≈ A*B
    end

    @testset "gaius" begin
        @test gaius_mul!(Y, A, B) ≈ A*B
        @test gaius_mul!(Y, A, B; multithreaded=false) ≈ A*B
    end
end
