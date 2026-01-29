# Credit: Post by stevengj
# https://discourse.julialang.org/t/julia-matrix-multiplication-performance/55175/12
using LoopVectorization

function add_matmul_rec!(m,n,p, i0,j0,k0, C,A,B; threshold=256)
    if m+n+p <= threshold   # base case: naive matmult for sufficiently large matrices
        @avx for i = 1:m, k = 1:p
            c = zero(eltype(C))
            for j = 1:n
                @inbounds c += A[i0+i,j0+j] * B[j0+j,k0+k]
            end
            @inbounds C[i0+i,k0+k] += c
        end
    else
        m2 = m ÷ 2; n2 = n ÷ 2; p2 = p ÷ 2
        add_matmul_rec!(m2, n2, p2, i0, j0, k0, C, A, B)
        
        add_matmul_rec!(m-m2, n2, p2, i0+m2, j0, k0, C, A, B)
        add_matmul_rec!(m2, n-n2, p2, i0, j0+n2, k0, C, A, B)
        add_matmul_rec!(m2, n2, p-p2, i0, j0, k0+p2, C, A, B)
        
        add_matmul_rec!(m-m2, n-n2, p2, i0+m2, j0+n2, k0, C, A, B)
        add_matmul_rec!(m2, n-n2, p-p2, i0, j0+n2, k0+p2, C, A, B)
        add_matmul_rec!(m-m2, n2, p-p2, i0+m2, j0, k0+p2, C, A, B)
        
        add_matmul_rec!(m-m2, n-n2, p-p2, i0+m2, j0+n2, k0+p2, C, A, B)
    end
    return C
end

"""
    stevengj_mul!(C, A, B; threshold=256)

Performs `C = A*B + C`, using divide-and-conquer, and LoopVectorization.jl. Reference: https://discourse.julialang.org/t/julia-matrix-multiplication-performance/55175/12
"""
stevengj_mul!(C, A, B; threshold=256) = add_matmul_rec!(size(A,1),size(A,2),size(B,2), 0,0,0, C,A,B, threshold=threshold)
