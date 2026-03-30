push!(LOAD_PATH, ".")
import Pkg
Pkg.activate(".")

using Plots
using LaTeXStrings


function cb(n,k)
    result = 1
    for i in 0:k-1
        result = result*(n-i)/(i+1)
    end
    return result
end

function sig(r)
    δ = 0.0001
    return sqrt(r^2 + δ^2) - δ
end

function step(σ)
    temp = 0
    if σ <= b
        for k in 0:n
            temp = temp + cb(n+k,k)*cb(2n+1,n-k)*(-σ/b)^k
        end
        temp = temp*(σ/b)^(n+1)
    else
        temp = 1.
    end

    return temp
end

function d_step(σ)
    temp = 0
    if σ <= b
        for k in 0:n
            temp = temp + cb(n+k,k)*cb(2n+1,n-k)*(-1)^k*(σ/b)^(n+k)*(n+k+1)/b
        end
    end
    return temp
end


range = -3:0.01:3
h  = zeros(length(range))
∂h = zeros(length(range))

rc = 2.
b = sig(rc)
n = 2

for (i,r) in enumerate(range)
    h[i]  = step(sig(r))
    ∂h[i] = d_step(sig(r))
end

gr(size=(500,400))
fig = plot(range, h, linestyles = :solid, linewidth=3,
                     legendfontsize = 16, tickfontsize = 20, 
                     framestyle = "", label = "", 
                     xlims = (minimum(range), maximum(range)), ylims = (-0.1, 1.2),
                     xticks = ([-rc,0,rc],["\$-b\$","\$0\$","\$b\$"]), 
                     yticks = ([0,1],["\$0\$","\$1\$"]))
plot!(range, ∂h, linestyles = :solid, linewidth=3, label = "")
annotate!(-1.0, 1.1, text("\${h(σ(\\Vert\\!\\mathbf{x}\\Vert_2))}\$", :red, :right, 20)),
annotate!( 3.0, 0.2, text("\${  {∂h}/{∂σ} }\$", :red, :right, 20)),

annotate!(2.9, -.19, text("\${\\mathbf{x}}\$", :red, :right, 20)) 
savefig(fig, joinpath("figs","Smooth_step_2D.pdf"))



Z = [step(sig(sqrt(r₁^2+r₂^2))) for r₁ in range, r₂ in range]


gr(size=(500,400))
fig = plot(range, range, Z, st=:surface,
            xticks = ([-rc,0,rc],["\$-b\$","\$0\$","\$b\$"]),
            yticks = ([-rc,0,rc],["\$-b\$","\$0\$","\$b\$"]))
annotate!(-1., -1., 1.1, text("\${h(σ(‖\\mathbf{x}‖_2))}\$", :red, :right, 20))
annotate!( 1., -1.8, 0, text("\${\\mathbf{x}}\$", :red, :right, 20))

savefig(fig, joinpath("figs","Smooth_step_3D.pdf"))