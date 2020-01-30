include("kernel.jl")


"""
Multivariate normal distribution
"""
function mvnorm(μ::AbstractVector{T}, K::AbstractMatrix{W}; seed::Integer=-1) where {T<:Real, W<:Real}
    U,S,V = svd(K)
    rng = MersenneTwister()
    if seed >= 0 
        rng = MersenneTwister(seed)
    end
    x = randn(rng,size(K)[1])
    y = μ .+ U*Diagonal(sqrt.(S))*x
end

"""
random sample by GP
"""
function sample(k::Kernel, xx::AbstractVector; seed::Integer=-1)
    K = kernel_matrix(k, xx)
    N = size(xx)[end]
    mvnorm(zeros(N), K; seed=seed) 
end



"""
Gaussian Process
"""
mutable struct GPKernel <: Kernel
    k::Kernel
end

mutable struct GaussianProcess 
    gpk::GPKernel
    η::Float64
end
GaussianProcess(k::Kernel) = GaussianProcess(GPKernel(k),0.0)
GaussianProcess(k::Kernel, η::Real) = GaussianProcess(GPKernel(k),exp(η))
GaussianProcess(k::GPKernel) = GaussianProcess(k,0.0)
GaussianProcess(k::GPKernel, η::Real) = GaussianProcess(k,exp(η))

function _predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, K::Matrix{S}, Kinv::AbstractMatrix{S}) where {T<:Real, S<:Real}
    s = kernel_matrix(gp.gpk.k, xtest) + [gp.η]
    k = kernel_matrix(gp.gpk.k, xtrain, xtest)
    μ = k'*Kinv*ytrain
    σ2 = s - k'*Kinv*k
    return μ,σ2
end
_predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, K::Matrix{S}, Kinv::AbstractMatrix{S}) where {T<:Real, S<:Real} = _predict(gp, [xtest], xtrain, ytrain, K, Kinv)

function predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}) where {T<:Real}
    μs = zero(xtest)
    σs = zero(xtest)
    K = kernel_matrix(gp.gpk.k, xtrain) + diagm(gp.η*ones(length(xtrain)))
    Kinv = Symmetric(inv(K))
    for (i,x) in enumerate(xtest)
        μ, σ2 = _predict(gp, x, xtrain, ytrain, K, Kinv)
        μs[i], σs[i] = μ[1], σ2[1]
    end
    σs[findall(x->x<0, σs)] .= 0.;
    return μs, σs
end
predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector,
        ytrain::AbstractVector{T}) where {T<:Real} = predict(gp, [xtest], xtrain, ytrain)