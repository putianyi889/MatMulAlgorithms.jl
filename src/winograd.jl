# Implementation of Winograd's modification of Strassen's algorithm.
# Reference: https://en.wikipedia.org/wiki/Strassen_algorithm#Improvements_to_Strassen_algorithm

"""
    winograd_mul!(Y, A, B; plan::Vector{StrassenStage{T}}, stageind) where T

Performs `Y = A*B + Y` using Winograd's modification of Strassen's algorithm. `plan` and `stageind` are optional. When the arguments have small sizes, the fallback method is [`winograd_mul_fallback!`](@ref) which by default redirect to `LinearAlgebra._mul!`.

!!! note
    This method is not numerically accurate. Use it only for exact algebra.
"""
function winograd_mul!(Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; plan::Vector{StrassenStage{T}} = strassen_plan(T, size(A,1), size(A,2), size(B,2)), stageind=1) where T
    if stageind > length(plan)
        return winograd_mul_fallback!(Y, A, B)
    end
    stage = plan[stageind]
    A11 = @view A[1:stage.m, 1:stage.k]
    A12 = @view A[1:stage.m, stage.k+1:2*stage.k]
    A21 = @view A[stage.m+1:2*stage.m, 1:stage.k]
    A22 = @view A[stage.m+1:2*stage.m, stage.k+1:2*stage.k]
    B11 = @view B[1:stage.k, 1:stage.n]
    B12 = @view B[1:stage.k, stage.n+1:2*stage.n]
    B21 = @view B[stage.k+1:2*stage.k, 1:stage.n]
    B22 = @view B[stage.k+1:2*stage.k, stage.n+1:2*stage.n]
    Y11 = @view Y[1:stage.m, 1:stage.n]
    Y12 = @view Y[1:stage.m, stage.n+1:2*stage.n]
    Y21 = @view Y[stage.m+1:2*stage.m, 1:stage.n]
    Y22 = @view Y[stage.m+1:2*stage.m, stage.n+1:2*stage.n]
    fill!(stage.TY, zero(T))
    winograd_mul!(stage.TY, A11, B11; plan=plan, stageind=stageind+1) # t
    Y11 .+= stage.TY
    winograd_mul!(Y11, A12, B21; plan=plan, stageind=stageind+1)
    stage.TA .= A21.+A22.-A11
    stage.TB .= B11.+B22.-B12
    winograd_mul!(stage.TY, stage.TA, stage.TB; plan=plan, stageind=stageind+1) # w
    Y21 .+= stage.TY
    Y12 .+= stage.TY
    Y22 .+= stage.TY
    stage.TA .= A21.-A11
    stage.TB .= B12.-B22
    fill!(stage.TY, zero(T))
    winograd_mul!(stage.TY, stage.TA, stage.TB; plan=plan, stageind=stageind+1) # u
    Y21 .+= stage.TY
    Y22 .+= stage.TY
    stage.TA .= A21.+A22
    stage.TB .= B12.-B11
    fill!(stage.TY, zero(T))
    winograd_mul!(stage.TY, stage.TA, stage.TB; plan=plan, stageind=stageind+1) # v
    Y12 .+= stage.TY
    Y22 .+= stage.TY
    stage.TA .= A11.+A12.-A21.-A22
    stage.TB .= B21.+B12.-B11.-B22
    winograd_mul!(Y12, stage.TA, B22; plan=plan, stageind=stageind+1)
    winograd_mul!(Y21, A22, stage.TB; plan=plan, stageind=stageind+1)
    if stage.mtail
        @views mul!(Y[end, :], transpose(B), A[end, :], true, true)
        if stage.ntail
            @views mul!(Y[1:end-1, end], A[1:end-1, :], B[:, end], true, true)
        end
    elseif stage.ntail
        @views mul!(Y[:, end], A, B[:, end], true, true)
    end
    return Y
end

"""
    winograd_mul_fallback!(Y, A, B)

The fallback method of [`winograd_mul!`](@ref). Override this for better tuning.
"""
winograd_mul_fallback!(Y, A, B) = LinearAlgebra._mul!(Y, A, B, true, true)
