using LinearAlgebra
using Printf
using Random
using SpecialFunctions

abstract type AbstractKernel end
abstract type Kernel <: AbstractKernel end


"""
Gaussian Kernel
"""
mutable struct GaussianKernel <: Kernel
    θ1::Float64
    θ2::Float64
end
GaussianKernel(θ1::Real, θ2::Real) = GaussianKernel(Float64(θ1), Float64(θ2))

kernel(k::GaussianKernel, x1::Real, x2::Real) = exp(k.θ1) * exp(-(x1 - x2)^2 / exp(k.θ2))

function kernel(k::GaussianKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    exp(k.θ1) * exp(-sum((x1 - x2).^2) / exp(k.θ2))
end

# function logderiv(gp::GaussianProcess, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
#     _logderiv(gp, gp.k, x1, x2)
# end
# logderiv(gp::GaussianProcess, x::AbstractVector{T}) where {T<:Real} = _logderiv(gp, gp.k, x, x)

# function update!(gp::GaussianProcess, grad::AbstractVector; lr=1e-3)
#     # θ1 = exp(τ)
#     gp.k.θ1 += lr*exp(grad[1])
#     gp.k.θ2 += lr*exp(grad[2])
#     gp.σ2 += lr*exp(grad[3])
# end


"""
Linear Kernel
"""
mutable struct LinearKernel <: Kernel
    θ::Float64
end
LinearKernel(θ::Real) = LinearKernel(Float64(θ))

kernel(k::LinearKernel, x1::Real, x2::Real) = x1 + x2 + k.θ

function kernel(k::LinearKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S <:Real}
    dot(x1,x2) + k.θ
end



"""
Exponential Kernel
"""
mutable struct ExponentialKernel <: Kernel
    θ::Float64
    function ExponentialKernel(θ::Float64)
        @assert θ != 0
        new(θ)
    end
end
ExponentialKernel(θ::Real) = ExponentialKernel(Float64(θ))

kernel(k::ExponentialKernel, x1::Real, x2::Real) = exp(-sum(abs(x1-x2)) / k.θ)

function kernel(k::ExponentialKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    exp(-sum(abs.(x1-x2)) / k.θ)
end


"""
Periodic Kernel
"""
mutable struct PeriodicKernel <: Kernel
    θ1::Float64
    θ2::Float64
    function PeriodicKernel(θ1::Float64, θ2::Float64)
        @assert θ1 != 0 && θ2 != 0
        new(θ1, θ2)
    end
end
PeriodicKernel(θ1::Real, θ2::Real) = PeriodicKernel(Float64(θ1), Float64(θ2))

kernel(k::PeriodicKernel, x1::Real, x2::Real) = exp(k.θ1 * cos(abs(x1-x2) / k.θ2))

function kernel(k::PeriodicKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    exp(k.θ1 * cos(sum(abs.(x1-x2)) / k.θ2))
end



"""
Matern3 Kernel
"""
mutable struct MaternKernel <: Kernel
    ν::Float64
    θ::Float64
    function MaternKernel(ν::Float64, θ::Float64)
        @assert θ != 0
        new(ν,θ)
    end
end
MaternKernel(ν::Real, θ::Real) = MaternKernel(Float64(ν), Float64(θ))
Matern3Kernel(θ::Real) = MaternKernel(3/2, Float64(θ))
Matern5Kernel(θ::Real) = MaternKernel(5/2, Float64(θ))

function kernel(k::MaternKernel, x1::Real, x2::Real)
    if x1 == x2
        return 1.0
    end
    r = abs(x1 - x2)
    t = sqrt(2 * k.ν) * r / k.θ
    2^(1-k.ν) / gamma(k.ν) * t^k.ν * besselk(k.ν, t)
end

function kernel(k::MaternKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    if x1 == x2
        return 1.0
    end
    r = sum(abs.(x1 - x2))
    t = sqrt(2 * k.ν) * r / k.θ
    2^(1-k.ν) / gamma(k.ν) * t^k.ν * besselk(k.ν, t)
end

mutable struct ConstantKernel <: Kernel
    θ::Float64
    function ConstantKernel(θ::Float64)
        @assert θ != 0
        new(θ)
    end
end
ConstantKernel() = ConstantKernel(1.0)
ConstantKernel(θ::Real) = ConstantKernel(Float64(θ))

kernel(k::ConstantKernel, x1::Real, x2::Real) = 1.0

function kernel(k::ConstantKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    length(x1) == length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    return 1.0
end


"""
Kernel Matrix
"""
function kernel_matrix(k::Kernel, xs1::AbstractVector, xs2::AbstractVector)
    N1 = length(xs1)
    N2 = length(xs2)
    K = zeros(Float64,N1,N2)
    for i in 1:N1
        for j in 1:N2
            K[i,j] = kernel(k, xs1[i], xs2[j])
        end
    end
    K
end

kernel_matrix(k::Kernel, x::AbstractVector) = kernel_matrix(k, x, x)




"""
Kernel Operation
"""

mutable struct KernelProduct <: Kernel
    coef::Float64
    k::Vector{T} where {T <: Kernel}
end
KernelProduct(k::Kernel) = KernelProduct(1.0, [k])
KernelProduct(coef::Real, k::Kernel) = KernelProduct(Float64(coef), [k])
KernelProduct(coef::Real, k::Vector{T}) where {T<:Kernel} = KernelProduct(Float64(coef), k)

Base.length(kp::KernelProduct) = 1 + sum([Base.length(k) for k in kp.kernel])

function kernel(kp::KernelProduct, x1, x2)
   kp.coef * prod([kernel(k, x1, x2) for k in kp.k])
end

Base.:*(coef::Real, k::Kernel) = KernelProduct(coef, k)
Base.:*(k::Kernel, coef::Real) = coef * k
Base.:*(coef::Real, kp::KernelProduct) = KernelProduct(Float64(coef)*k.coef, kp.k)
Base.:*(kp::KernelProduct, coef::Real) = coef * kp

Base.:*(k1::Kernel, k2::Kernel) = KernelProduct(1.0, [k1, k2])
Base.:*(kp::KernelProduct, k::Kernel) = KernelProduct(kp.coef, [kp.kernel..., k])
Base.:*(k::Kernel, kp::KernelProduct) = kp * k
Base.:*(kp1::KernelProduct, kp2::KernelProduct) = KernelProduct(kp1.coef * kp2.coef, [kp1.kernel..., kp2.kernel...])

Base.:-(k::Kernel) = -1.0 * k
Base.:-(kp::KernelProduct) = -1.0 * kp



mutable struct KernelSum <: Kernel
    k::Vector{KernelProduct}
end

Base.length(ks::KernelSum) = sum([Base.length(k) for k in ks.kernel])

function kernel(ks::KernelSum, x1, x2)
    sum([kernel(k, x1, x2) for k in ks.k])
end

Base.:*(coef::Real, ks::KernelSum) = KernelSum([coef * k for k in ks.k])
Base.:*(ks::KernelSum, coef::Real) = coef * ks

Base.:+(kp1::KernelProduct, kp2::KernelProduct) = KernelSum([kp1, kp2])
Base.:+(ks::KernelSum, kp::KernelProduct) = KernelSum([ks.k..., kp])
Base.:+(kp::KernelProduct, ks::KernelSum) = ks + kp

Base.:+(k1::Kernel, k2::Kernel) = KernelProduct(k1) + KernelProduct(k2)
Base.:+(k1::AbstractKernel, k2::Kernel) = k1 + KernelProduct(k2)
Base.:+(k1::Kernel, k2::AbstractKernel) = k2 + k1

Base.:-(k1::AbstractKernel, k2::AbstractKernel) = k1 + (-k2)



"""
log derivative
"""
# function logderiv(gp::GaussianProcess, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
#     _logderiv(gp, gp.k, x1, x2)
# end
# logderiv(gp::GaussianProcess, x::AbstractVector{T}) where {T<:Real} = _logderiv(gp, gp.k, x, x)


"""
Likelihood
"""
# function loglikelihood(gp::GaussianProcess, xtrain::AbstractVector{T},
#         ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
#     K = kernel_matrix(gp.k, xtrain)
#     Kinv = inv(K)
#     -log(det(K)) - ytrain' * Kinv * ytrain
# end

# function loglikelihood_deriv(Kgrad)
#     -tr(Kinv * Kgrad) + transpose(Kinv * ytrain) * Kgrad * (Kinv * ytrain)
# end

