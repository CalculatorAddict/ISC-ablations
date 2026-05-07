#!/usr/bin/env julia
#=
  Oja's Rule — Time Scale Separation Experiment
  =============================================

  Explores how the ratio between input fluctuation rate and weight
  dynamics rate affects the asymptotics of Oja's learning rule.

  Two control parameters:
    η  — learning rate (faster weight dynamics ↔ larger η)
    ρ  — AR(1) autocorrelation of inputs (slower fluctuations ↔ larger ρ)

  The mean-field approximation is valid when η · τ_corr ≪ 1, where
  τ_corr = 1/(1−ρ) is the input autocorrelation time.  This script
  maps out the (η, ρ) plane and compares stochastic trajectories
  against the deterministic mean-field ODE and its logistic reduction.

  Usage:  julia oja_experiment.jl
  Deps:   using Pkg; Pkg.add(["Plots", "LaTeXStrings", "Printf"])
=#

# import Pkg; Pkg.add("Plots"); Pkg.add("LaTeXStrings")

using LinearAlgebra, Statistics, Random, Printf
using Plots, LaTeXStrings
gr()

# ═══════════════════════════════════════════════════════════════════
# Problem setup
# ═══════════════════════════════════════════════════════════════════

const DIM     = 2
const EIGVALS = Float64[5,3]
const COV     = Diagonal(EIGVALS)               # C = diag(λ₁,…,λₙ)
const COVSQ   = Diagonal(sqrt.(EIGVALS))         # C^{1/2}
const E1      = vcat(1.0, zeros(DIM - 1))        # principal eigenvector

# ═══════════════════════════════════════════════════════════════════
# AR(1) correlated inputs
#   x_{t+1} = ρ x_t + √(1−ρ²) C^{1/2} z_t,   z_t ∼ N(0,I)
#   stationary covariance  E[x x⊤] = C
# ═══════════════════════════════════════════════════════════════════

function generate_inputs!(X::Matrix{Float64}, ρ::Float64, rng::AbstractRNG)
    n, T = size(X)
    x = COVSQ * randn(rng, n)
    σ = sqrt(max(1.0 - ρ^2, 0.0))
    for t in 1:T
        x .= ρ .* x .+ σ .* (COVSQ * randn(rng, n))
        X[:, t] .= x
    end
end

# ═══════════════════════════════════════════════════════════════════
# Stochastic Oja (discrete-time)
#   w ← w + η (y x − y² w),   y = w⊤ x
# ═══════════════════════════════════════════════════════════════════

function oja_discrete!(w::Vector{Float64}, X::Matrix{Float64}, η::Float64,
                       alignment::AbstractVector{Float64},
                       norm_sq::AbstractVector{Float64})
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
        alignment[t] = abs(dot(w, E1)) / sqrt(s)
    end
end

# ═══════════════════════════════════════════════════════════════════
# Mean-field ODE:  dw/dt = η (C w − (w⊤C w) w)
# Integrated with RK4, recorded at integer time steps.
# ═══════════════════════════════════════════════════════════════════

function meanfield_ode(w0::Vector{Float64}, η::Float64, T::Int;
                       substeps::Int = 20)
    w  = copy(w0)
    dt = 1.0 / substeps
    alignment = zeros(T)
    norm_sq   = zeros(T)

    k1 = similar(w); k2 = similar(w); k3 = similar(w); k4 = similar(w)
    tmp = similar(w)

    function rhs!(out, u)
        Cu  = COV * u
        wCw = dot(u, Cu)
        @. out = η * (Cu - wCw * u)
    end

    for t in 1:T
        for _ in 1:substeps
            rhs!(k1, w)
            @. tmp = w + 0.5 * dt * k1;   rhs!(k2, tmp)
            @. tmp = w + 0.5 * dt * k2;   rhs!(k3, tmp)
            @. tmp = w + dt * k3;          rhs!(k4, tmp)
            @. w  += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        end
        norm_sq[t]   = dot(w, w)
        alignment[t] = abs(dot(w, E1)) / sqrt(norm_sq[t])
    end
    return alignment, norm_sq
end

# ═══════════════════════════════════════════════════════════════════
# Logistic analytical solution for ‖w‖²
#   z(t) = z₀ eʳᵗ / (1 − z₀ + z₀ eʳᵗ),   r = 2ηλ
# ═══════════════════════════════════════════════════════════════════

function logistic_norm(z0::Float64, λ::Float64, η::Float64, T::Int)
    r = 2.0 * η * λ
    return [z0 * exp(r * t) / (1.0 - z0 + z0 * exp(r * t)) for t in 1:T]
end

# ═══════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════

function run_experiment(ηs, ρs, T, nruns; seed = 42)
    results = Dict{Tuple{Float64,Float64}, NamedTuple}()
    X = zeros(DIM, T)
    total = length(ηs) * length(ρs)
    cnt = 0

    for η in ηs, ρ in ρs
        cnt += 1
        @printf("\r  [%2d/%d]  η = %-6.3f  ρ = %-5.2f", cnt, total, η, ρ)

        all_align = zeros(nruns, T)
        all_norm  = zeros(nruns, T)

        for r in 1:nruns
            # Same w₀ for each run index across all (η, ρ)
            rng_w = MersenneTwister(seed + r)
            w0 = randn(rng_w, DIM)
            w0 ./= norm(w0)

            # Input sequence depends on (η, ρ, r)
            rng_x = MersenneTwister(seed + 1000 * r + hash((η, ρ)) % 100_000)
            generate_inputs!(X, ρ, rng_x)

            w = copy(w0)
            oja_discrete!(w, X, η, view(all_align, r, :), view(all_norm, r, :))
        end

        valid = .!isnan.(all_align[:, end])
        results[(η, ρ)] = (
            align_mean  = vec(mean(all_align[valid, :], dims = 1)),
            align_std   = vec(std(all_align[valid, :],  dims = 1)),
            norm_mean   = vec(mean(all_norm[valid, :],  dims = 1)),
            norm_std    = vec(std(all_norm[valid, :],   dims = 1)),
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
# axis. Linearly extrapolates outside the data range so the boundary curve
# can be drawn out to the diagram edges.
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

# Overlay the stability boundary η̂ λ₁ = 2  (i.e. η = c (1−ρ), c = 2/λ₁).
# The curve is sampled wide and clipped to the diagram bounds [0.5, N+0.5],
# i.e. to the outer edges of the boundary cells rather than their centres.
function add_stability_line!(p, ηs, ρs; c = 0.4, npts = 1000, kwargs...)
    xlo, xhi = 0.5, length(ρs) + 0.5
    ylo, yhi = 0.5, length(ηs) + 0.5
    xs, ys = Float64[], Float64[]
    for ρ in range(first(ρs) - 1.0, last(ρs) + 1.0, length = npts)
        η_t = c * (1 - ρ)
        xi  = _cat_pos(ρs, ρ)
        yi  = _cat_pos(ηs, η_t)
        (xlo <= xi <= xhi) || continue
        (ylo <= yi <= yhi) || continue
        push!(xs, xi)
        push!(ys, yi)
    end
    plot!(p, xs, ys; color = :white, lw = 3, ls = :solid,
          label = L"\hat\eta\lambda_1 = 2", kwargs...)
end

# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

function fig_phase_diagram(res, ηs, ρs)
    Z = [mean(res[(η, ρ)].final_align) for η in ηs, ρ in ρs]
    heatmap(string.(ρs), string.(ηs), Z;
        xlabel = L"\rho", ylabel = L"\eta",
        title  = L"\mathrm{Mean\ final\ alignment}\ \langle |w^\top e_1|/\|w\| \rangle",
        color = :viridis, clims = (0.0, 1.0),
        size = (620, 440), margin = 5Plots.mm)
end

function fig_fluctuation_map(res, ηs, ρs)
    Z = [std(res[(η, ρ)].final_align) for η in ηs, ρ in ρs]
    heatmap(string.(ρs), string.(ηs), Z;
        xlabel = L"\rho", ylabel = L"\eta",
        title  = L"\mathrm{Std\ of\ final\ alignment}",
        color = :inferno,
        size = (620, 440), margin = 5Plots.mm)
end

function fig_stability_map(res, ηs, ρs, nruns)
    Z = [res[(η, ρ)].n_valid / nruns for η in ηs, ρ in ρs]
    p = heatmap(1:length(ρs), 1:length(ηs), Z;
        xticks = (1:length(ρs), string.(ρs)),
        yticks = (1:length(ηs), string.(ηs)),
        xlabel = L"\rho", ylabel = L"\eta",
        title  = "Fraction of non-divergent runs",
        color = :RdYlGn, clims = (0.0, 1.0),
        size = (620, 440), margin = 5Plots.mm,
        legend = false)
    add_stability_line!(p, ηs, ρs)
    p
end

function fig_convergence_vs_eta(res, ηs, ρ_fix, T)
    p = plot(; xlabel = "Iteration", ylabel = L"|w^\top e_1|/\|w\|",
        title  = latexstring("\\mathrm{Alignment\\ convergence,\\ } \\rho=$ρ_fix"),
        legend = :bottomright, size = (660, 420))
    for η in ηs
        r = res[(η, ρ_fix)]
        plot!(p, 1:T, r.align_mean; ribbon = r.align_std,
            label = latexstring("\\eta=$η"), lw = 2, alpha = 0.6)
    end
    p
end

function fig_convergence_vs_rho(res, η_fix, ρs, T)
    p = plot(; xlabel = "Iteration", ylabel = L"|w^\top e_1|/\|w\|",
        title  = latexstring("\\mathrm{Alignment\\ convergence,\\ } \\eta=$η_fix"),
        legend = :bottomright, size = (660, 420))
    for ρ in ρs
        r = res[(η_fix, ρ)]
        plot!(p, 1:T, r.align_mean; ribbon = r.align_std,
            label = latexstring("\\rho=$ρ"), lw = 2, alpha = 0.6)
    end
    p
end

function fig_norm_dynamics(res, η, ρ, T, nruns; seed = 42, ylims = nothing)
    mf_norms = zeros(nruns, T)
    for r in 1:nruns
        rng_w = MersenneTwister(seed + r)
        w0 = randn(rng_w, DIM); w0 ./= norm(w0)
        _, mf_norms[r, :] = meanfield_ode(w0, η, T)
    end
    mf_mean = vec(mean(mf_norms, dims = 1))

    r = res[(η, ρ)]
    p = plot(; xlabel = "Iteration", ylabel = L"\|\mathbf w\|^2",
        title  = latexstring("\\mathrm{Norm\\ dynamics,\\ } \\eta=$η,\\; \\rho=$ρ"),
        legend = :topright, size = (660, 420))
    ylims !== nothing && ylims!(p, ylims)
    plot!(p, 1:T, r.norm_mean; ribbon = r.norm_std, color = 1,
        label = "Stochastic (mean ± std)", lw = 2, alpha = 0.4)
    plot!(p, 1:T, mf_mean; color = 2,
        label = "Mean-field ODE", lw = 2, ls = :dash)
    hline!(p, [1.0]; color = :gray, lw = 1, ls = :dot, label = L"\|w\|^2 = 1")
    p
end

# Compute shared y-limits across the pairs you want linked
function shared_ylims(res, configs; pad = 0.05)
    lo, hi = Inf, -Inf
    for (η, ρ) in configs
        r = res[(η, ρ)]
        lo = min(lo, minimum(r.norm_mean .- r.norm_std))
        hi = max(hi, maximum(r.norm_mean .+ r.norm_std))
    end
    span = hi - lo
    (lo - pad*span, hi + pad*span)
end

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

function main()
    ηs    = [0.001, 0.003, 0.01, 0.03, 0.05]
    ρs    = [0.0, 0.3, 0.6, 0.9, 0.95, 0.99]
    T     = 5000
    nruns = 30

    println("═══════════════════════════════════════════════════")
    println("  Oja's Rule — Time Scale Separation Experiment")
    println("═══════════════════════════════════════════════════")
    println("  dim = $DIM,   T = $T,   nruns = $nruns")
    println("  eigenvalues = $EIGVALS")
    println("  η  ∈ $ηs")
    println("  ρ  ∈ $ρs")
    println()

    # ── Run stochastic simulations ────────────────────────────
    t0  = time()
    res = run_experiment(ηs, ρs, T, nruns)
    @printf("  Simulation time: %.1fs\n\n", time() - t0)

    # ── Generate and save figures ─────────────────────────────
    mkpath("figures")
    println("  Saving figures...")

    savefig(fig_phase_diagram(res, ηs, ρs),
            "figures/phase_diagram.pdf")
    savefig(fig_fluctuation_map(res, ηs, ρs),
            "figures/fluctuation_map.pdf")
    savefig(fig_stability_map(res, ηs, ρs, nruns),
            "figures/stability_map.pdf")

    savefig(fig_convergence_vs_eta(res, ηs, 0.0, T),
            "figures/conv_eta_rho00.pdf")
    savefig(fig_convergence_vs_eta(res, ηs, 0.9, T),
            "figures/conv_eta_rho09.pdf")

    savefig(fig_convergence_vs_rho(res, 0.003, ρs, T),
            "figures/conv_rho_eta003.pdf")
    savefig(fig_convergence_vs_rho(res, 0.03,  ρs, T),
            "figures/conv_rho_eta030.pdf")

    slow_pairs = [(0.003, 0.0), (0.003, 0.9)]
    fast_pairs = [(0.03,  0.0), (0.03,  0.9)]

    slow_lims = shared_ylims(res, slow_pairs)
    fast_lims = shared_ylims(res, fast_pairs)

    savefig(fig_norm_dynamics(res, 0.003, 0.0, T, nruns; ylims = slow_lims),
            "figures/norm_slow_iid.pdf")
    savefig(fig_norm_dynamics(res, 0.003, 0.9, T, nruns; ylims = slow_lims),
            "figures/norm_slow_corr.pdf")
    savefig(fig_norm_dynamics(res, 0.03,  0.0, T, nruns; ylims = fast_lims),
            "figures/norm_fast_iid.pdf")
    savefig(fig_norm_dynamics(res, 0.03,  0.9, T, nruns; ylims = fast_lims),
            "figures/norm_fast_corr.pdf")

    println("  All figures saved to figures/\n")

    # ── Summary table ─────────────────────────────────────────
    println("  ── Summary ──────────────────────────────────────")
    @printf("  %-7s  %-5s  %8s  %8s  %7s\n",
            "eta", "rho", "<align>", "std", "valid")
    println("  " * "─"^42)
    for η in ηs, ρ in ρs
        r = res[(η, ρ)]
        @printf("  %-7.3f  %-5.2f  %8.4f  %8.4f  %3d/%d\n",
                η, ρ, mean(r.final_align), std(r.final_align), r.n_valid, nruns)
    end
    println()
end

main()