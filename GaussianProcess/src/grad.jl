abstract type Grad end

mutable struct GaussianKernelGrad <: Grad
    gp::GaussianProcess
    K::Matrix{Float64}
    K_inv::Matrix{Float64}
end
GaussianKernelGrad(gp::GaussianProcess, K::Matrix{Real}, K_inv::Matrix{Real}) = GaussianKernelGrad(gp, Float64.(K), Float64.(K_inv))

function grad(gpg::GaussianKernelGrad, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    N = length(x1)
    K_noise = diagm(ones(N).* gpg.gp.σ2) 
    
    dkdτ = gpg.K .- K_noise
    dkdσ = (gpg.K .- K_noise) .* .-(x1, x2').^2 ./ gpg.gp.k.θ2
    [dkdτ, dkdσ]
end

mutable struct LinearKernelGrad <: Grad
    gp::GaussianProcess
    K::Matrix{Float64}
    K_inv::Matrix{Float64}
end
LinearKernelGrad(gp::GaussianProcess, K::Matrix{Real}, K_inv::Matrix{Real}) = LinearKernelGrad(gp, Float64.(K), Float64.(K_inv))

function grad(gpg::LinearKernelGrad, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    N = length(x1)
    K_noise = diagm(ones(N).* gpg.gp.σ2) 
    
    dkdτ = gpg.K .- K_noise
    [dkdτ]
end

# mutable struct ExponentialKernelGrad <: Grad
#     gp::GaussianProcess
#     K::Matrix{Float64}
#     K_inv::Matrix{Float64}
# end
# ExponentialKernelGrad(gp::GaussianProcess, K::Matrix{Real}, K_inv::Matrix{Real}) = ExponentialKernelGrad(gp, Float64.(K), Float64.(K_inv))

# mutable struct PeriodicKernelGrad <: Grad
#     gp::GaussianProcess
#     K::Matrix{Float64}
#     K_inv::Matrix{Float64}
# end
# PeriodicKernelGrad(gp::GaussianProcess, K::Matrix{Real}, K_inv::Matrix{Real}) = PeriodicKernelGrad(gp, Float64.(K), Float64.(K_inv))

# mutable struct MaternKernelGrad <: Grad
#     gp::GaussianProcess
#     K::Matrix{Float64}
#     K_inv::Matrix{Float64}
# end
# MaternKernelGrad(gp::GaussianProcess, K::Matrix{Real}, K_inv::Matrix{Real}) = MaternKernelGrad(gp, Float64.(K), Float64.(K_inv))

"""
Likelihood
"""
function loglikelihood(gkg::Grad, xtrain::AbstractVector{T},
        ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
    -log(det(gkg.K)) - ytrain' * gkg.K_inv * ytrain
end

function loglikelihood_grad(gpg::Grad, Kgrad::AbstractMatrix{T}, ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
    -tr(gpg.K_inv * Kgrad) + transpose(gpg.K_inv * ytrain) * Kgrad * (gpg.K_inv * ytrain)
end

