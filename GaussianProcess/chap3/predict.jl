include("../src/kernel.jl")
include("../src/gp.jl")

using Plots
pyplot()
using Plots.PlotMeasures
using LaTeXStrings

xtrain = vcat(rand(80),rand(20) * 2 .+ 1.0)
xtest = collect(range(0,stop=4,length=1000))
ytrain = sin.(π.*xtrain) #+ randn(length(xtrain)).*0.1
ytest = sin.(π.*xtest)

k = GaussianKernel(1.0, 0.4) + DiagonalConstantKernel(0.1)
gp = GaussianProcess(k)
μs,σs = predict(gp, xtest, xtrain, ytrain)


p = plot(xtest, μs-2sqrt.(σs), label="±2σ", alpha=0,
    fill=μs+2sqrt.(σs), fillalpha=0.3, color=:red)
plot!(xtrain, ytrain, st=:scatter, label="train", ms=5, color=:blue)
plot!(xtest, ytest, st=:line, label="truth", color=:black)
plot!(xtest, μs, st=:line, label="gp", lw=3, ls=:dash, color=:red)

savefig("predict.png")