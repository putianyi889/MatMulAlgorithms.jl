# MatMulAlgorithms

[![Build Status](https://github.com/putianyi889/MatMulAlgorithms.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/putianyi889/MatMulAlgorithms.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://putianyi889.github.io/MatMulAlgorithms.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://putianyi889.github.io/MatMulAlgorithms.jl/dev)

The generic routine provided by Julia Base is naive and slow. This package provides a collection of fast matrix multiplication routines that excel in certain cases.

The routines do not override the generic routines in LinearAlgebra. The user may customize their own choice by something like
```julia
import LinearAlgebra.mul!

function mul!(Y::MyMatrix, A::MyMatrix, B::MyMatrix, α, β)
    if isone(α)
        some_routine!(Y, A, B, β)
    end
end
```

Note that the routines usually don't check the dimensions of the matrices. 
