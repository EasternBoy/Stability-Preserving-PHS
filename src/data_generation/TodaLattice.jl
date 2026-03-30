# Toda lattice from Yuwei Geng, Data-Driven Reduced-Order Models for Port-Hamiltonian Systems with Operator Inference
# https://arxiv.org/pdf/2501.02183
push!(LOAD_PATH, ".")
import Pkg
Pkg.activate(".")

using DiffEqCallbacks, OrdinaryDiffEq, LinearAlgebra, Plots, NPZ, LinearAlgebra, SparseArrays
using Dates


# 1: step signal, 2: pulse signal, 3: sinusoid signal, 
global input_type = 1
global mode = "train"
train_time = 1000
test_time  = 50


mutable struct TodaLattice
    n_cells::Int64
    s::Vector{Float64}
    γ::Vector{Float64}
    ϵ::Float64
    ω::Float64

    function TodaLattice(n_cells::Int64, s::Vector{Float64}, γ::Vector{Float64})
        ϵ = 1.
        ω = 1.
        return new(n_cells, s, γ, ϵ, ω)
    end
end


function PHSystem(ds, s, config, t)
    N = config.n_cells
    u = input(t)
    γ = config.γ
    ϵ = config.ϵ
    ω = config.ω

    Jₚ = [spzeros(N,N) sparse(I, N, N); -sparse(I, N, N) spzeros(N,N)]
    Rₚ = [spzeros(N,N) spzeros(N,N); spzeros(N,N) spdiagm(γ)]
    Gₚ = [spzeros(N)' 1. spzeros(N-1)']'

    ∇H = spzeros(2N)
    for i in 1:N
        if i == 1
            ∇H[i]   = exp(s[i] - s[i+1]) - 1 + ϵ*ω*sin(ω*s[i])
        elseif i==N
            ∇H[i]   = exp(s[i]) - exp(s[i-1] - s[i])
        else
            ∇H[i]   = exp(s[i] - s[i+1]) - exp(s[i-1] - s[i])
        end
        ∇H[i+N] = s[i+N]
    end

    ds[1:2N] = Vector((Jₚ - Rₚ)*∇H + Gₚ*u)
end



function step(t)
    if t >= 5
        return 0.5
    else
        return 0.
    end
end

function pulse(t)
    if t >= 5 && t <= 20
        return 0.5
    else
        return 0.
    end
end

function mod_sin(t; period = 10, amp = 0.5)
    return amp*sin(2pi*t/period)
end

func = Vector{Function}([step, pulse, mod_sin])
name = ["step", "pulse", "sin"]

function input(t::Float64, mode = mode, input_type = input_type)
    global pre_u
    if mode == "train"
        if t%rand([0.5 1. 2. 3. 4. 5.]) == 0
            pre_u = rand() - rand()
        end
        return pre_u + 1e-2rand(1)[1]
    else
        return func[input_type](t)
    end
end


function ODEcallback(s, t, integrator)
    config = integrator.p
    N = config.n_cells
    u = input(t)
    γ = config.γ
    ϵ = config.ϵ
    ω = config.ω

 
    Jₚ = [spzeros(N,N) sparse(I, N, N); -sparse(I, N, N) spzeros(N,N)]
    Rₚ = [spzeros(N,N) spzeros(N,N);     spzeros(N,N)    spdiagm(γ)]
    Gₚ = [spzeros(N)' 1. spzeros(N-1)']'


    H  = 0
    ∇H = spzeros(2N)
    for i in 1:N
        if i == 1
            ∇H[i]   = exp(s[i] - s[i+1]) - 1 + ϵ*ω*sin(ω*s[i])
            H      += (1/2)*s[i+N]^2 + exp(s[i] - s[i+1]) + ϵ*(1-cos(ω*s[i]))
        elseif i == N
            ∇H[i]   = exp(s[i]) - exp(s[i-1] - s[i])
        else
            ∇H[i]   = exp(s[i] - s[i+1]) - exp(s[i-1] - s[i])
            H  += (1/2)*s[i+N]^2 + exp(s[i] - s[i+1])
        end
        ∇H[i+N] = s[i+N]
    end

    H += exp(s[N]) - s[1] - N
    ds = Vector((Jₚ - Rₚ)*∇H + Gₚ*u)
    y  = Gₚ'*∇H
    return s[1:end], ds, H, ∇H, y, u
end


function ODEsolver(obj::TodaLattice, sampling)
    s0           = obj.s
    prob         = ODEProblem(PHSystem, s0, (sampling[1], sampling[end]), obj);
    saved_values = SavedValues(Float64, Tuple{Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}, Float64, Float64});
    cb           = SavingCallback(ODEcallback, saved_values, saveat = sampling);
    sol          = solve(prob, BS5(), callback = cb, saveat = sampling);
    obj.s        = sol.u[end];
    return saved_values, sol
end

# Julia 1.12 world-age safe dispatch when this file is reloaded under Revise.
run_odesolver(obj::TodaLattice, sampling) = Base.invokelatest(ODEsolver, obj, sampling)


t, d = now(), today()
date_time = string(month(d),"-",day(d),"_",hour(t),":",minute(t))


n_cells = 5
γ       = 0.5*ones(n_cells)
initial = zeros(2n_cells)
toda    = TodaLattice(n_cells, initial, γ)


global mode = "train"
global pre_u = 0.

sampling_train = 0.:0.1:train_time
sampling_test  = 0.:0.1:test_time


#Select basic vectors
#Reduce order r
#If r = 2n_cells, full order options.
if mode == "train"
    saved_values, sol = run_odesolver(toda, sampling_train)
    ns    = length(saved_values.t)
    Xs    = reduce(hcat, [saved_values.saveval[i][1] for i in 1:ns])
    Xdots = reduce(hcat, [saved_values.saveval[i][2] for i in 1:ns])
    Hs    = [saved_values.saveval[i][3] for i in 1:ns]
    ∇Hs   = reduce(hcat, [saved_values.saveval[i][4] for i in 1:ns])
    y     = [saved_values.saveval[i][5] for i in 1:ns]
    u     = [saved_values.saveval[i][6] for i in 1:ns]

    r       = 2n_cells
    V, Λ, U = svd(Xs)
    V       = V[:,1:r]

    path = joinpath("data", "TodaLat_data_train.npz")
    npzwrite(path, Dict(
        "Xs"     => Xs,
        "Xdots"  => Xdots,
        "Ys"     => y,
        "Hs"     => Hs,
        "gradHs" => ∇Hs,
        "Us"     => u,
        "Ts"     => saved_values.t,
        "Vtrans" => V))
end

fig = plot()
plot!(sampling_train, y, label="y")
plot!(sampling_train, u, label="u")
savefig(fig, joinpath("data", "TodaLat_y_u_" * mode * ".png"))


global mode = "test"
for type in 1:3
    global input_type = type
    global pre_u = 0.

    toda.s = zeros(2n_cells)
    saved_values, sol = run_odesolver(toda, sampling_test)
    ns    = length(saved_values.t)
    Xs    = reduce(hcat, [saved_values.saveval[i][1] for i in 1:ns])
    Xdots = reduce(hcat, [saved_values.saveval[i][2] for i in 1:ns])
    Hs    = [saved_values.saveval[i][3] for i in 1:ns]
    ∇Hs   = reduce(hcat, [saved_values.saveval[i][4] for i in 1:ns])
    y     = [saved_values.saveval[i][5] for i in 1:ns]
    u     = [saved_values.saveval[i][6] for i in 1:ns]

    r       = 2n_cells
    V, Λ, U = svd(Xs)
    V       = V[:,1:r]

    path = joinpath("data", string(name[type], "_",  "TodaLat_data_test.npz"))
    npzwrite(path, Dict(
        "Xs"     => Xs,
        "Xdots"  => Xdots,
        "Ys"     => y,
        "Hs"     => Hs,
        "gradHs" => ∇Hs,
        "Us"     => u,
        "Ts"     => saved_values.t,
        "Vtrans" => V))
    fig = plot()
    plot!(sampling_test, y, label="y")
    plot!(sampling_test, u, label="u")
    savefig(fig, joinpath("data", "TodaLat_y_u_" * mode * name[type] *".png"))
end
