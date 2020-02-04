include("../src/kernel.jl")
include("../src/gp.jl")

using Printf
using Plots
pyplot(fmt=:png)
using Plots.PlotMeasures
using LaTeXStrings
import DelimitedFiles: readdlm
 
path = @sprintf("%s/gpr.dat", @__FILE__)
train = readdlm(path, '\t', Float64, '\n')
xtrain = train[:,1]
ytrain = train[:,2]
xtest = collect(range(-1,stop=4,length=100))

# GP parameters
τ = 1.
σ = 2.
η = 0.1

k = GaussianKernel(τ, σ)
gp = GaussianProcess(k, η)
μs,σs = predict(gp, xtest, xtrain, ytrain)

p = plot(xtest, μs-2sqrt.(σs); label="±2σ", alpha=0, fill=μs+2sqrt.(σs), fillalpha=0.3, color=:red)
plot!(xtrain, ytrain; st=:scatter, label="train", ms=5, color=:blue)
plot!(xtest, μs; st=:line, label="gp", lw=3, ls=:dash, color=:red)

savepath = @sprintf("%s/predict.png", @__DIR__)
savefig(savepath)