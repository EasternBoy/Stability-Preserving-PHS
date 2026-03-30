push!(LOAD_PATH,".")
import Pkg
Pkg.activate(".")

using DiffEqCallbacks, OrdinaryDiffEq
using LinearAlgebra
using Plots, NPZ, Random
using ForwardDiff

dt = 0.05
train_time = 10.0
test_time  = 20.0

n_train_traj = 100
n_test_traj  = 1

sampling_train = 0:dt:train_time
sampling_test  = 0:dt:test_time

mutable struct DoublePendulum
    s::Vector{Float64}
    γ::Vector{Float64}
    m1::Float64
    m2::Float64
    l1::Float64
    l2::Float64
    g::Float64
end

function MassMatrix(q, config)
    θ1, θ2 = q
    m1, m2 = config.m1, config.m2
    l1, l2 = config.l1, config.l2

    Δ = θ1 - θ2

    [
        (m1+m2)*l1^2     m2*l1*l2*cos(Δ)
        m2*l1*l2*cos(Δ)  m2*l2^2
    ]
end

function Hamiltonian(x, config)
    θ1, θ2, p1, p2 = x

    q = [θ1, θ2]
    p = [p1, p2]

    M = MassMatrix(q, config)

    T = 0.5 * p' * inv(M) * p

    m1, m2 = config.m1, config.m2
    l1, l2 = config.l1, config.l2
    g = config.g

    V = -(m1+m2)*g*l1*cos(θ1) - m2*g*l2*cos(θ2)

    return T + V
end


function gradH(x, config)
    return ForwardDiff.gradient(s -> Hamiltonian(s, config), x)
end


function PHSystem(ds, s, config, t)
    γ = config.γ

    J = [
        0  0  1  0
        0  0  0  1
       -1  0  0  0
        0 -1  0  0
    ]

    R = [
        0  0  0     0
        0  0  0     0
        0  0  γ[1]  0
        0  0  0     γ[2]
    ]

    ∇H = gradH(s, config)

    ds[:] = (J - R) * ∇H
end


function ODEcallback(s, t, integrator)
    config = integrator.p

    ∇H = gradH(s, config)
    H  = Hamiltonian(s, config)

    γ = config.γ

    J = [
        0  0  1  0
        0  0  0  1
       -1  0  0  0
        0 -1  0  0
    ]

    R = [
        0  0  0     0
        0  0  0     0
        0  0  γ[1]  0
        0  0  0     γ[2]
    ]

    ds = (J - R) * ∇H

    y = [s[1], s[2]]

    return copy(s), ds, copy(H), ∇H, y
end

function simulate(obj, sampling)
    prob = ODEProblem(PHSystem, obj.s, (sampling[1], sampling[end]), obj)

    saved_values = SavedValues(Float64,
        Tuple{Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}, Vector{Float64}})

    cb = SavingCallback(ODEcallback, saved_values, saveat=sampling)

    sol = solve(prob, Tsit5(), callback=cb, saveat=sampling)

    return saved_values
end

function random_initial()
    θ1 = 0.0 #rand()*(10π) - 5π
    θ2 = 0.0 #rand()*(10π) - 5π

    p1 = randn()*10 -5
    p2 = randn()*10 -5

    [θ1, θ2, p1, p2]
end

m1 = 2.0
m2 = 2.0
l1 = 0.5
l2 = 0.5
g  = 9.81

γ  = [0.5, 0.5]


Xk_train_vec   = Vector{Float64}[]
Xkp1_train_vec = Vector{Float64}[]

grid_size = 10
p1_values = range(-10, 10, length=grid_size)
p2_values = range(-2, 2, length=grid_size)

for p1 in p1_values
    for p2 in p2_values
        θ1 = 0.0
        θ2 = 0.0
        initial = [θ1, θ2, p1, p2]
        pend = DoublePendulum(initial, γ, m1, m2, l1, l2, g)
        saved = simulate(pend, sampling_train)
        ns = length(saved.t)
        states = [saved.saveval[i][1] for i in 1:ns]
        for k in 1:(ns-1)
            xk   = states[k]
            xkp1 = states[k+1]
            push!(Xk_train_vec, xk)
            push!(Xkp1_train_vec, xkp1)
        end
    end
end

Xk_train   = reduce(hcat, Xk_train_vec)
Xkp1_train = reduce(hcat, Xkp1_train_vec)

mkpath("data")

npzwrite("data/double_pendulum_train_dt$(dt).npz",
    Dict(
        "Xk" => Xk_train,
        "Xkp1" => Xkp1_train
    )
)

Xk_test_vec   = Vector{Float64}[]
Xkp1_test_vec = Vector{Float64}[]
H_test_vec = Float64[]

for traj in 1:n_test_traj
    initial = random_initial()
    initial = [0, 0, 10, -2]
    pend = DoublePendulum(initial, γ, m1, m2, l1, l2, g)
    saved = simulate(pend, sampling_test)
    
    ns = length(saved.t)
    states = [saved.saveval[i][1] for i in 1:ns]
    Hs = [saved.saveval[i][3] for i in 1:ns]
    for k in 1:(ns-1)
        xk   = states[k]
        xkp1 = states[k+1]
        push!(Xk_test_vec, xk)
        push!(Xkp1_test_vec, xkp1)
        push!(H_test_vec, Hs[k])
    end
end

Xk_test   = reduce(hcat, Xk_test_vec)
Xkp1_test = reduce(hcat, Xkp1_test_vec)
H = collect(H_test_vec)

npzwrite("data/double_pendulum_testX01_dt$(dt).npz",
    Dict(
        "Xk" => Xk_test,
        "Xkp1" => Xkp1_test,
        "H"     => H,
    )
)



