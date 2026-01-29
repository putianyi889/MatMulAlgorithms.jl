
"""
    StrassenStage{T}

A pre-allocated buffer for one stage of Strassen multiplication for eltype `T`.

Let the multiplication be `Y = A * B` where `Y` is `M × N`, `A` is `M × K` and `B` is `K × N`. In Strassen's algorithm, all 3 matrices are evenly partitioned into 4 blocks. If some of the dimensions are odd, the last row/column is handled separately, and are flagged by `mtail`, `ktail` and `ntail`. 

# Fields
- `mtail, ktail, ntail::Bool`.
- `m, k, n::Int`: sizes of partitioned blocks. `M = 2m + mtail` and so on.
- `TY, TA, TB::Matrix{T}`: Buffers. `TY` is `m × k`, `TA` is `k × n`, `TB` is `m × n`.
"""
struct StrassenStage{T}
    m::Int
    k::Int
    n::Int
    TY::Matrix{T}
    TA::Matrix{T}
    TB::Matrix{T}
    mtail::Bool
    ktail::Bool
    ntail::Bool
end

"""
    strassen_plan(T, m, k, n; threshold = 64)

Generates a vector of [`StrassenStage`](@ref)s for the Strassen multiplication of `m × k` and `k × n` matrices of eltype `T`. The threshold determines the minimum size of the matrices that are multiplied directly.
"""
function strassen_plan(::Type{T}, m, k, n; threshold = 64) where T
    plan = StrassenStage{T}[]
    while m*n*k >= threshold^3
        m2 = m ÷ 2
        k2 = k ÷ 2
        n2 = n ÷ 2
        push!(plan, StrassenStage{T}(m2, k2, n2, similar(Matrix{T}, m2, n2), similar(Matrix{T}, m2, k2), similar(Matrix{T}, k2, n2), isodd(m), isodd(k), isodd(n)))
        m = m2
        k = k2
        n = n2
    end
    return plan
end
