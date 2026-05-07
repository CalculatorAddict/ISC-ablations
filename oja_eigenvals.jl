#!/usr/bin/env julia
#=
  Oja's Rule — Eigenvalue Grid Experiment
  ========================================

  Sweeps (λ₁, λ₂) with fixed η and ρ, plotting stability maps.
  Cells where λ₂ > λ₁ are masked (not a valid covariance spectrum).
=#

using LinearAlgebra, Statistics, Random, Printf
using Plots, LaTeXStrings
gr()

const DIM = 2

function generate_inputs!(X::Matrix{Float64}, ρ::Float64,
                          covsq::Diagonal, rng::AbstractRNG)
    n, T = size(X)
    x = covsq * randn(rng, n)
    σ = sqrt(max(1.0 - ρ^2, 0.0))
    for t in 1:T
        x .= ρ .* x .+ σ .* (covsq * randn(rng, n))
        X[:, t] .= x
    end
end

function oja_discrete!(w::Vector{Float64}, X::Matrix{Float64}, η::Float64,
                       alignment::AbstractVector{Float64},
                       norm_sq::AbstractVector{Float64},
                       e1::Vector{Float64})
    T = size(X, 2)
    for t in 1:T
        xt = @view X[:, t]
        y  = dot(w, xt)
        @. w += η * (y * xt - y^2 * w)
        s = dot(w, w)
        if !isfinite(s)
            alignment[t:end] .= NaN
            norm_sq[t:end]   .= NaN
            return
        end
        norm_sq[t]   = s
        alignment[t] = abs(dot(w, e1)) / sqrt(s)
    end
end

function run_eigenvalue_grid(λ1s, λ2s, η, ρ, T, nruns; seed=42)
    results = Dict{Tuple{Float64,Float64}, Union{NamedTuple, Nothing}}()
    X = zeros(DIM, T)
    total = length(λ1s) * length(λ2s)
    cnt   = 0

    for λ1 in λ1s, λ2 in λ2s
        cnt += 1
        @printf("\r  [%2d/%d]  λ₁ = %-5g  λ₂ = %-5g", cnt, total, λ1, λ2)

        if λ2 > λ1
            results[(λ1, λ2)] = nothing
            continue
        end

        eigvals = Float64[λ1, λ2]
        covsq   = Diagonal(sqrt.(eigvals))
        e1      = vcat(1.0, zeros(DIM - 1))

        all_align = zeros(nruns, T)
        all_norm  = zeros(nruns, T)

        for r in 1:nruns
            rng_w = MersenneTwister(seed + r)
            w0    = rand(rng_w, DIM) .- 0.5

            rng_x = MersenneTwister(seed + 1000*r + hash((λ1, λ2)) % 100_000)
            generate_inputs!(X, ρ, covsq, rng_x)

            w = copy(w0)
            oja_discrete!(w, X, η, view(all_align, r, :), view(all_norm, r, :), e1)
        end

        valid = .!isnan.(all_align[:, end])
        results[(λ1, λ2)] = (
            align_mean  = vec(mean(all_align[valid, :], dims=1)),
            norm_mean   = vec(mean(all_norm[valid,  :], dims=1)),
            final_align = all_align[valid, end],
            n_valid     = sum(valid),
        )
    end
    println("\r  Done.                                          ")
    return results
end

# ═══════════════════════════════════════════════════════════════════
# Plotting helpers (used only by the stability map)
# ═══════════════════════════════════════════════════════════════════

# Map a continuous value into the index position used by a categorical-style
# axis. Linearly extrapolates outside the data range.
function _cat_pos(values, target)
    n = length(values)
    if target < values[1]
        return 1.0 - (values[1] - target) / (values[2] - values[1])
    elseif target > values[end]
        return float(n) + (target - values[end]) / (values[end] - values[end-1])
    end
    for i in 1:n-1
        if values[i] <= target <= values[i+1]
            return i + (target - values[i]) / (values[i+1] - values[i])
        end
    end
    return NaN
end

# Overlay η̂ λ₁ = 2 in the (λ₂, λ₁) plane. With η, ρ fixed this is a
# horizontal line at λ₁ = 2(1−ρ)/η, clipped to the valid covariance
# region (λ₂ ≤ λ₁_crit).
function add_stability_line!(p, λ1s, λ2s, η, ρ; kwargs...)
    λ1_crit = 2 * (1 - ρ) / η
    yi = _cat_pos(λ1s, λ1_crit)
    ylo, yhi = 0.5, length(λ1s) + 0.5
    (ylo <= yi <= yhi) || return

    xlo = 0.5
    xhi = length(λ2s) + 0.5
    xhi <= xlo && return

    plot!(p, [xlo, xhi], [yi, yi]; color = :white, lw = 3, ls = :solid,
          label = "", kwargs...)
end

# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

function fig_stability_grid(res, λ1s, λ2s, nruns, η, ρ)
    Z = [isnothing(res[(λ1, λ2)]) ? NaN :
         res[(λ1, λ2)].n_valid / nruns
         for λ1 in λ1s, λ2 in λ2s]

    p = heatmap(1:length(λ2s), 1:length(λ1s), Z;
        xticks = (1:length(λ2s), string.(λ2s)),
        yticks = (1:length(λ1s), string.(λ1s)),
        xlabel  = L"\lambda_2",
        ylabel  = L"\lambda_1",
        title   = latexstring(
            "\\mathrm{Fraction\\ stable,\\ }\\eta=$(η),\\ \\rho=$(ρ)"),
        color   = :RdYlGn,
        clims   = (0.0, 1.0),
        size    = (600, 460),
        margin  = 5Plots.mm,
        legend  = false)
    add_stability_line!(p, λ1s, λ2s, η, ρ)
    p
end

function fig_alignment_grid(res, λ1s, λ2s, η, ρ)
    Z = [isnothing(res[(λ1, λ2)]) ? NaN :
         (isempty(res[(λ1, λ2)].final_align) ? NaN :
          mean(res[(λ1, λ2)].final_align))
         for λ1 in λ1s, λ2 in λ2s]

    heatmap(string.(λ2s), string.(λ1s), Z;
        xlabel  = L"\lambda_2",
        ylabel  = L"\lambda_1",
        title   = latexstring(
            "\\mathrm{Mean\\ final\\ alignment,\\ }\\eta=$(η),\\ \\rho=$(ρ)"),
        color   = :viridis,
        clims   = (0.0, 1.0),
        size    = (600, 460),
        margin  = 5Plots.mm)
end

function main()
    λ1s = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    λ2s = [0.0, 0.5, 1.0, 2.0, 3.0]
    T     = 5000
    nruns = 30
    η     = 0.05
    ρ     = 0.95

    println("═══════════════════════════════════════════════════")
    println("  Oja's Rule — Eigenvalue Grid Experiment")
    println("═══════════════════════════════════════════════════")
    println("  dim = $DIM,   T = $T,   nruns = $nruns")
    println("  η = $η,   ρ = $ρ")
    println("  λ₁ ∈ $λ1s")
    println("  λ₂ ∈ $λ2s")
    println()

    t0  = time()
    res = run_eigenvalue_grid(λ1s, λ2s, η, ρ, T, nruns)
    @printf("  Simulation time: %.1fs\n\n", time() - t0)

    mkpath("figures/oja-eigenvals")
    savefig(fig_stability_grid(res, λ1s, λ2s, nruns, η, ρ),
            "figures/oja-eigenvals/stability_eigenvals.pdf")
    savefig(fig_alignment_grid(res, λ1s, λ2s, η, ρ),
            "figures/oja-eigenvals/alignment_eigenvals.pdf")

    @printf("\n  %-6s  %-6s  %8s  %8s  %7s\n",
            "lam1", "lam2", "<align>", "std", "valid")
    println("  " * "─"^44)
    for λ1 in λ1s, λ2 in λ2s
        r = res[(λ1, λ2)]
        if isnothing(r)
            @printf("  %-6g  %-6g  %8s  %8s  %s\n",
                    λ1, λ2, "—", "—", "masked (λ₂>λ₁)")
        else
            fa = r.final_align
            @printf("  %-6g  %-6g  %8.4f  %8.4f  %3d/%d\n",
                    λ1, λ2, isempty(fa) ? NaN : mean(fa),
                    isempty(fa) ? NaN : std(fa), r.n_valid, nruns)
        end
    end

    println("\n  All figures saved to figures/")
end

main()