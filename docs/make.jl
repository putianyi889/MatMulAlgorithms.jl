using Documenter
using MatMulAlgorithms

makedocs(
    sitename = "MatMulAlgorithms",
    format = Documenter.HTML(),
    modules = [MatMulAlgorithms]
)

deploydocs(
    repo = "github.com/putianyi889/MatMulAlgorithms.jl"
)
