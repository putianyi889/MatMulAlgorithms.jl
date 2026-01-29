# MatMulAlgorithms.jl

The generic routine provided by Julia Base is naive and slow. This package provides a collection of fast matrix multiplication routines that excel in certain cases.

The routines do not override the generic routines in LinearAlgebra. The user may customize their own choice by something like
```julia
import LinearAlgebra.mul!

function mul!(Y::MyMatrix, A::MyMatrix, B::MyMatrix, α, β)
    if isone(α)
        some_routine!(Y, A, B)
    end
end
```

Note that the routines usually don't check the dimensions of the matrices. Some routines add to existing values of `Y` and some overwrite them.

Supported routines:
- [`stevengj_mul!`](@ref)
- [`winograd_mul!`](@ref)
- [`octavian_mul!`](@ref)
- [`gaius_mul!`](@ref)
