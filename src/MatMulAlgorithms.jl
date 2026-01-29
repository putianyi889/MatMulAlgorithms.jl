module MatMulAlgorithms

using LinearAlgebra
import Octavian
import Gaius

export stevengj_mul!, winograd_mul!, octavian_mul!, gaius_mul!

include("stevengj.jl")
include("strassenplan.jl")
include("winograd.jl")

"""
    octavian_mul!(C, A, B)

Performs `C = A*B`. This is a wrapper around [Octavian.jl](https://github.com/JuliaLinearAlgebra/Octavian.jl).
"""
octavian_mul!(C, A, B) = Octavian.matmul!(C, A, B)

"""
    gaius_mul!(C, A, B; multithreaded=true)

Performs `C = A*B`. This is a wrapper around [Gaius.jl](https://github.com/MasonProtter/Gaius.jl).
"""
function gaius_mul!(C, A, B; multithreaded=true)
    if multithreaded
        Gaius.mul!(C, A, B)
    else
        Gaius.mul_serial!(C, A, B)
    end
end

end
