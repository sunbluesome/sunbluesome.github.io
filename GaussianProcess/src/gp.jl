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
mutable struct GaussianProcess 
    k::AbstractKernel
end

function _predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, K::Matrix{S}) where {T<:Real, S<:Real}
    N = size(K,1)
    K_inv = Symmetric(inv(K))
    s = kernel_matrix(gp.k,xtest)
    k = kernel_matrix(gp.k,xtrain, xtest)
    μ = k'*K_inv*ytrain
    σ2 = s - k'*K_inv*k
    return μ,σ2
end
_predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, K::Matrix{S}) where {T<:Real, S<:Real} = _predict(gp, [xtest], xtrain, ytrain, K)

function predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}) where {T<:Real}
    μs = zero(xtest)
    σs = zero(xtest)
    K = kernel_matrix(gp.k, xtrain)
    for (i,x) in enumerate(xtest)
        μ, σ2 = _predict(gp, x, xtrain, ytrain, K)
        μs[i], σs[i] = μ[1], σ2[1]
    end
    σs[findall(x->x<0, σs)] .= 0.;
    return μs, σs
end
predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector,
        ytrain::AbstractVector{T}) where {T<:Real} = predict(gp, [xtest], xtrain, ytrain)