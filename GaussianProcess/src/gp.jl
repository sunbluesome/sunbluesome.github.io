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
    k::Kernel
    η::Float64
    function GaussianProcess(k::Kernel, η::Float64)
        new(k, η)
    end
end
GaussianProcess(k::Kernel) = GaussianProcess(k,0.0)
GaussianProcess(k::Kernel, η::Real) = GaussianProcess(k,Float64(η))

function logderiv(gp::GaussianProcess, x1::AbstractVector, x2::AbstractVector)
    klogderiv = logderiv(gp.k, x1, x2)
    nlogderiv = diagm(gp.η*ones(length(x1)))
    return [klogderiv..., nlogderiv]
end
logderiv(gp::GaussianProcess, xx::AbstractVector) = logderiv(gp, xx, xx)

function _predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, K::Matrix{S}, Kinv::AbstractMatrix{S}) where {T<:Real, S<:Real}
    s = kernel_matrix(gp.k, xtest) + [gp.η]
    k = kernel_matrix(gp.k, xtrain, xtest)
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
    K = kernel_matrix(gp.k, xtrain) + diagm(gp.η*ones(length(xtrain)))
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

function update!(gp::GaussianProcess, params::Vector{T}) where {T<:Real}
    update!(gp.k, params[1:end-1])
    gp.η = params[end]
end


"""
Likelihood
"""
function loglikelihood(gp::GaussianProcess, xtrain::AbstractVector{T},
        ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
    K = kernel_matrix(gp.k, xtrain) + diagm(gp.η*ones(length(xtrain)))
    Kinv = Symmetric(inv(K))
    detK = det(K)
    if detK <= 0
        return missing
    else
        return -log(det(K)) - ytrain' * Kinv * ytrain
    end
end

function loglikelihood_partialderiv(gp::GaussianProcess, ytrain::AbstractVector{S},
        Kinv::Matrix{U}, Kgrad::Matrix{U}) where {T<:Real, S<:Real, U<:Real}
    -tr(Kinv * Kgrad) + transpose(Kinv * ytrain) * Kgrad * (Kinv * ytrain)
end

function loglikelihood_deriv(gp::GaussianProcess, xtrain::AbstractVector{T},
        ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
    K = kernel_matrix(gp.k, xtrain) + diagm(gp.η*ones(length(xtrain)))
    Kinv = Matrix(Symmetric(inv(K)))
    Kinvy = Kinv * ytrain
    Kgrads = logderiv(gp, xtrain)
    [loglikelihood_partialderiv(gp, ytrain, Kinv, Kgrad) for Kgrad in Kgrads]
end
