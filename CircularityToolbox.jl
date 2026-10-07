"""
    CircularityToolbox

Embeds a hierarchical circularity index (OCI = sum of weighted SubIndices,
each a weighted sum of Indicator ratios N/D) in JuMP models.

Hierarchy
    Indicator   one ratio N/D, with a weight within its SubIndex
    SubIndex    group of Indicators, with a weight within the OCI
    OCI         = sum_j w_j * SubIndex_j
                = sum_k ω_k * N_k(x,y)/D_k(x,y)

The OCI is decomposed as
    OCI = L(x,y) + sum_{g=1}^G N_g(x,y) / D_g(x,y),

where:
    L(x,y) collects all indicators whose denominator is constant on 
    the feasible region;

    N_g(x,y) is the weighted aggregate numerator of variable-denominator
    group g;

    D_g(x,y) is the common variable denominator of group g;

    G is the number of distinct variable denominators.

Classification
    A   G = 0
        All denominators are constant. OCI is affine and requires
        no reformulation.

    B   G = 1 and K₀ = ∅
        There is a single variable denominator. The ratios can be aggregated
        and cross-multiplied.

    C   otherwise
        Either multiple distinct variable denominators exist
        (G >= 2), or a variable denominator coexists with
        constant-denominator indicators (G >= 1 and K₀ != ∅).

        The OCI requires either:
            strategy = :exact
                exact bilinear reformulation;

            strategy = :mccormick
                McCormick relaxation

Workflow
    1. Define Indicators and group them into SubIndices. A raw metric I_m(x) is
       min-max normalized to [0,1] with `normalized_indicator` (bounds computed
       automatically, or given); a metric that is already a ratio N/D uses
       `Indicator` directly.
    2. Analyze:
           A = analyze([si1, si2, ...], model)
    3. Add OCI >= tau:
           h = add_circularity!([si1, si2, ...], model; tau = 0.6)
    4. Solve:
           optimize!(model)
    5. Verify:
           verify(h)
"""
module CircularityToolbox

using JuMP
import MathOptInterface as MOI

export Indicator, SubIndex, Norm_Indicator, analyze, add_circularity!, remove_circularity!, verify,
       optimize_circularity!,
       sweep_tau, Analysis, CircularityHandle, CircularityReport, flatten

# ---------------------------------------------------------------------------
# 1. Indicator / SubIndex hierarchy
# ---------------------------------------------------------------------------

_aff(x::Number) = AffExpr(Float64(x))
_aff(x::VariableRef) = 1.0 * x
_aff(x::AffExpr) = x
_aff(x) = error("Numerator/denominator must be affine (number, variable or affine expression); got $(typeof(x)).")

"""
    Indicator(name, num, den; weight = 1.0, lower_better = false)

One ratio `num/den`. `weight` is relative to the other metrics WITHIN the same
`SubIndex` (normalized to sum to 1 when the SubIndex is analyzed). Both `num`
and `den` must be affine in the model variables. If `lower_better = true` the
metric enters as `1 - num/den = (den-num)/den`, so every metric is oriented
"higher = more circular".
"""
struct Indicator
    name::String
    num::AffExpr
    den::AffExpr
    weight::Float64
end

function Indicator(name::AbstractString, num, den; weight::Real = 1.0, lower_better::Bool = false)
    N, D = _aff(num), _aff(den)
    weight >= 0 || error("Weight of metric '$name' must be nonnegative.")
    if lower_better
        N = D - N
    end
    return Indicator(String(name), N, D, Float64(weight))
end

"""
    normalized_indicator(name, expr, model;
                         lo = :auto, hi = :auto,
                         lower_better = false, weight = 1.0,
                         verbose = false)

Build an `Indicator` from a RAW indicator `expr` (any affine expression I_m(x))
with min-max normalization, so that SI_m lies in [0, 1]:

    higher is better :  SI_m = (I_m - lo) / (hi - lo)
    lower  is better :  SI_m = 1 - (I_m - lo) / (hi - lo) = (hi - I_m) / (hi - lo)

`lo` / `hi` are I_m^min / I_m^max. Each can be

  * a number       -> used as given (e.g. a theoretical bound such as 0 or sum(CI_IN)),
  * `:auto`        -> computed as min / max of `expr` over the feasible set of
                      `model` (two LPs on the integrality-relaxed model).

The denominator `hi - lo` is a constant, so the indicator is affine in the model
variables and does not add any bilinear term. Call this BEFORE `analyze` /
`add_circularity!`, and before adding the circularity constraint itself.
The relaxed range is an outer approximation: SI_m is guaranteed to be in [0,1],
but it may not reach exactly 0 or 1 at any integer-feasible point.
"""
function Norm_Indicator(
    name::AbstractString,
    expr,
    model::Model;
    lo = :auto,
    hi = :auto,
    lower_better::Bool = false,
    weight::Real = 1.0,
    verbose::Bool = false
)
    # ============================================================
    # CASE 1: AFFINE RAW METRIC
    # ============================================================

    if _is_affine(expr)

        e = _aff(expr)

        if lo === :auto || hi === :auto
            rlo, rhi = _range(model, e)

            lo_v = lo === :auto ? rlo : Float64(lo)
            hi_v = hi === :auto ? rhi : Float64(hi)
        else
            lo_v = Float64(lo)
            hi_v = Float64(hi)
        end

        (isfinite(lo_v) && isfinite(hi_v)) ||
            error(
                "Indicator '$name': unbounded range " *
                "($lo_v, $hi_v). Pass finite `lo`/`hi`."
            )

        hi_v > lo_v ||
            error(
                "Indicator '$name': hi must be > lo " *
                "(got lo = $lo_v, hi = $hi_v)."
            )

        verbose && println(
            "  normalize ",
            rpad(name, 22),
            " lo = ", round(lo_v, sigdigits = 8),
            "   hi = ", round(hi_v, sigdigits = 8),
            lower_better ? "   (lower is better)" : "   (higher is better)"
        )

        return Indicator(
            name,
            e - lo_v,
            hi_v - lo_v;
            weight = weight,
            lower_better = lower_better
        )
    end

    # ============================================================
    # CASE 2: FRACTIONAL RAW METRIC N/D
    # ============================================================

    parts = _fraction_parts(expr)

    parts === nothing ||
        begin

            N, D = parts

            # ----------------------------------------------------
            # Automatic bounds
            # ----------------------------------------------------
            if lo === :auto || hi === :auto
                rlo, rhi = _fractional_range(model, N, D)

                lo_v = lo === :auto ? rlo : Float64(lo)
                hi_v = hi === :auto ? rhi : Float64(hi)
            else
                lo_v = Float64(lo)
                hi_v = Float64(hi)
            end

            # ----------------------------------------------------
            # Validate
            # ----------------------------------------------------
            (isfinite(lo_v) && isfinite(hi_v)) ||
                error(
                    "Indicator '$name': unbounded range " *
                    "($lo_v, $hi_v). Pass finite `lo`/`hi`."
                )

            hi_v > lo_v ||
                error(
                    "Indicator '$name': hi must be > lo " *
                    "(got lo = $lo_v, hi = $hi_v)."
                )

            verbose && println(
                "  normalize ",
                rpad(name, 22),
                " lo = ", round(lo_v, sigdigits = 8),
                "   hi = ", round(hi_v, sigdigits = 8),
                lower_better ? "   (lower is better)" : "   (higher is better)"
            )

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # Normalize N/D while keeping the result as an
            # affine-over-affine Indicator.
            # ----------------------------------------------------

            if lower_better

                # (hi - N/D) / (hi-lo)
                #
                # = (hi*D - N) / ((hi-lo)*D)

                normalized_num = hi_v * D - N

                normalized_den = (hi_v - lo_v) * D

            else

                # (N/D - lo) / (hi-lo)
                #
                # = (N - lo*D) / ((hi-lo)*D)

                normalized_num = N - lo_v * D

                normalized_den = (hi_v - lo_v) * D
            end

            return Indicator(
                name,
                normalized_num,
                normalized_den;
                weight = weight,
                lower_better = false
            )
        end

    # ============================================================
    # CASE 3: Unsupported nonlinear expression
    # ============================================================

    error(
        "Indicator '$name' is nonlinear but is not recognized as " *
        "an affine-over-affine fractional expression. " *
        "Automatic normalization currently supports affine metrics " *
        "and ratios N/D with affine numerator and denominator."
    )
end

"""
    SubIndex(name, metrics::Vector{Indicator}; weight = 1.0)

A named group of metrics representing one CE dimension (e.g. "energy",
"emissions"). `weight` is relative to the other sub-indices in the OCI
(normalized to sum to 1 across all sub-indices passed to `analyze` /
`add_circularity!`).
"""
struct SubIndex
    name::String
    metrics::Vector{Indicator}
    weight::Float64
end

function SubIndex(name::AbstractString, metrics::Vector{Indicator}; weight::Real = 1.0)
    isempty(metrics) && error("SubIndex '$name' has no metrics.")
    weight >= 0 || error("Weight of sub-index '$name' must be nonnegative.")
    return SubIndex(String(name), metrics, Float64(weight))
end

"""
Internal flat view of one metric with its FINAL weight in the OCI
(sub-index weight times metric weight, both normalized), plus provenance
for reporting (which sub-index it came from).
"""
struct _FlatMetric
    label::String          # "subindex.metric", for reports
    subindex::String
    name::String
    num::AffExpr
    den::AffExpr
    weight::Float64         # final weight in the OCI, sums to 1 over all _FlatMetric
end

"""
    flatten(ocis::Vector{SubIndex}) -> Vector{_FlatMetric}

Compute final OCI-level weights: normalize sub-index weights to sum to 1,
normalize each sub-index's metric weights to sum to 1, and multiply. This is
what `analyze`/`add_circularity!` operate on internally; exposed for
inspection and for printing supplementary-material tables.
"""
function flatten(ocis::Vector{SubIndex})
    Wtot = sum(si.weight for si in ocis)
    Wtot > 0 || error("All sub-index weights are zero.")
    flat = _FlatMetric[]
    for si in ocis
        wj = si.weight / Wtot
        wi_tot = sum(m.weight for m in si.metrics)
        wi_tot > 0 || error("All metric weights in sub-index '$(si.name)' are zero.")
        for m in si.metrics
            omega = m.weight / wi_tot
            push!(flat, _FlatMetric("$(si.name).$(m.name)", si.name, m.name, m.num, m.den, wj * omega))
        end
    end
    return flat
end

# ---------------------------------------------------------------------------
# helpers on affine expressions
# ---------------------------------------------------------------------------

function _is_const(e::AffExpr)
    d = copy(e)
    drop_zeros!(d)
    return isempty(d.terms)
end

function _same_aff(a::AffExpr, b::AffExpr; tol = 1e-9)
    d = a - b
    drop_zeros!(d)
    return isempty(d.terms) && abs(d.constant) <= tol
end
# Returns s > 0 such that a == s * b (within tol), or nothing.
function _prop_factor(a::AffExpr, b::AffExpr; tol = 1e-9)
    isempty(b.terms) && return nothing
    v = first(keys(b.terms))
    haskey(a.terms, v) || return nothing
    s = a.terms[v] / b.terms[v]
    s > 0 || return nothing
    return _same_aff(a, s * b; tol = tol) ? s : nothing
end
function _rescale!(flat, i, den_rep, s)
    m = flat[i]
    flat[i] = Indicator(
        m.name,
        m.num / s,
        den_rep,
        m.weight
    )
end
function _is_affine(expr)
    try
        _aff(expr)
        return true
    catch
        return false
    end
end

function _fraction_parts(expr)
    if expr isa NonlinearExpr
        # JuMP represents a/b as a NonlinearExpr with head :/
        if expr.head == :/ && length(expr.args) == 2
            num = expr.args[1]
            den = expr.args[2]

            try
                return _aff(num), _aff(den)
            catch
                return nothing
            end
        end
    end

    return nothing
end

function _fractional_range(
    model::Model,
    num::AffExpr,
    den::AffExpr;
    tol = 1e-8,
    max_iter = 60
)
    # ------------------------------------------------------------
    # 1. Bounds on numerator and denominator
    # ------------------------------------------------------------
    nlo, nhi = _range(model, num)
    dlo, dhi = _range(model, den)

    if !isfinite(nlo) || !isfinite(nhi)
        error(
            "Cannot determine finite bounds for the numerator of the " *
            "fractional indicator."
        )
    end

    if !isfinite(dlo) || !isfinite(dhi)
        error(
            "Cannot determine finite bounds for the denominator of the " *
            "fractional indicator."
        )
    end

    # The denominator must remain strictly positive.
    if dlo <= 0
        error(
            "Automatic normalization of a fractional indicator requires " *
            "the denominator to be strictly positive over the feasible region. " *
            "Obtained denominator lower bound = $dlo."
        )
    end

    # ------------------------------------------------------------
    # 2. Initial conservative bounds for num / den
    # ------------------------------------------------------------
    ratio_lo = nlo / dhi
    ratio_hi = nhi / dlo

    if ratio_lo > ratio_hi
        error(
            "Invalid initial bounds for fractional indicator: " *
            "[$ratio_lo, $ratio_hi]."
        )
    end

    # ------------------------------------------------------------
    # 3. Save current objective
    # ------------------------------------------------------------
    old_sense = objective_sense(model)
    old_obj   = objective_function(model)

    undo_relax = relax_integrality(model)

    # Initialize the returned bounds BEFORE entering try.
    rmin = ratio_lo
    rmax = ratio_hi

    try

        # --------------------------------------------------------
        # Feasibility test:
        #       num / den <= λ
        #
        # Since den > 0:
        #       num - λ den <= 0
        # --------------------------------------------------------
        function feasible_leq(λ)
            c = @constraint(model, num - λ * den <= 0)

            try
                set_objective_sense(model, MOI.FEASIBILITY_SENSE)
                optimize!(model)

                return termination_status(model) == MOI.OPTIMAL
            finally
                delete(model, c)
            end
        end

        # --------------------------------------------------------
        # Feasibility test:
        #       num / den >= λ
        #
        # Since den > 0:
        #       num - λ den >= 0
        # --------------------------------------------------------
        function feasible_geq(λ)
            c = @constraint(model, num - λ * den >= 0)

            try
                set_objective_sense(model, MOI.FEASIBILITY_SENSE)
                optimize!(model)

                return termination_status(model) == MOI.OPTIMAL
            finally
                delete(model, c)
            end
        end

        # --------------------------------------------------------
        # 4. Find minimum ratio
        # --------------------------------------------------------
        lo = ratio_lo
        hi = ratio_hi

        for _ in 1:max_iter
            mid = (lo + hi) / 2

            if hi - lo <= tol
                break
            end

            if feasible_leq(mid)
                # A feasible point exists with ratio <= mid.
                # Therefore the minimum is <= mid.
                hi = mid
            else
                # No feasible point has ratio <= mid.
                # Therefore the minimum is > mid.
                lo = mid
            end
        end

        # Use the conservative lower bound.
        rmin = lo

        # --------------------------------------------------------
        # 5. Find maximum ratio
        # --------------------------------------------------------
        lo = ratio_lo
        hi = ratio_hi

        for _ in 1:max_iter
            mid = (lo + hi) / 2

            if hi - lo <= tol
                break
            end

            if feasible_geq(mid)
                # A feasible point exists with ratio >= mid.
                # Therefore the maximum is >= mid.
                lo = mid
            else
                # No feasible point has ratio >= mid.
                # Therefore the maximum is < mid.
                hi = mid
            end
        end

        # Use the conservative upper bound.
        rmax = hi

    finally
        undo_relax()
        set_objective(model, old_sense, old_obj)
    end

    return Float64(rmin), Float64(rmax)
end

function _nonlinear_range(model::Model, expr)

    old_obj = objective_function(model)
    old_sense = objective_sense(model)

    undo_relax = relax_integrality(model)

    try
        # Use an NLP solver for nonlinear bound calculation
        set_optimizer(model, Ipopt.Optimizer)

        # Minimum
        set_objective_function(model, expr)
        set_objective_sense(model, MOI.MIN_SENSE)

        optimize!(model)

        if !is_solved_and_feasible(model)
            error(
                "Failed to compute lower bound for nonlinear indicator. " *
                "Termination status: $(termination_status(model))"
            )
        end

        lo = objective_value(model)

        # Maximum
        set_objective_sense(model, MOI.MAX_SENSE)

        optimize!(model)

        if !is_solved_and_feasible(model)
            error(
                "Failed to compute upper bound for nonlinear indicator. " *
                "Termination status: $(termination_status(model))"
            )
        end

        hi = objective_value(model)

    finally
        set_objective_function(model, old_obj)
        set_objective_sense(model, old_sense)

        undo_relax()
    end

    return Float64(lo), Float64(hi)
end

# ---------------------------------------------------------------------------
# 2. Range of an expression over the feasible set of the (relaxed) model
#    Two LPs: min and max of `expr` over the model with integrality relaxed.
#    The relaxation is an outer approximation, so the interval is valid but may be wide.
# ---------------------------------------------------------------------------

function _range(model::Model, expr::AffExpr)
    sense0 = objective_sense(model)
    obj0 = objective_function(model)
    undo = relax_integrality(model)
    lo, hi = -Inf, Inf
    try
        set_objective(model, MOI.MIN_SENSE, expr)
        optimize!(model)
        st = termination_status(model)
        if st == MOI.OPTIMAL
            lo = objective_value(model)
        elseif st != MOI.DUAL_INFEASIBLE
            @warn "LP for lower bound ended with status $st; bound left at -Inf"
        end
        set_objective(model, MOI.MAX_SENSE, expr)
        optimize!(model)
        st = termination_status(model)
        if st == MOI.OPTIMAL
            hi = objective_value(model)
        elseif st != MOI.DUAL_INFEASIBLE
            @warn "LP for upper bound ended with status $st; bound left at +Inf"
        end
    finally
        undo()
        set_objective(model, sense0, obj0)
    end
    return lo, hi
end

# ---------------------------------------------------------------------------
# 3. Analysis and classification
# ---------------------------------------------------------------------------

"""
Result of `analyze`.

  flat
      flattened metrics with their final OCI weight.

  case
      :A, :B or :C.

  const_idx
      indices into `flat` whose denominator is constant on the
      feasible set.

  groups
      indices into `flat` corresponding to variable-denominator
      metrics, grouped by identical denominator expression.

  den_range
      lower and upper bounds of each metric denominator.

  n_bilinear
      number of distinct variable denominator groups.

  remainder_const
      whether L(x,y) is a true constant. This is diagnostic only
      and does not determine the A/B/C classification.

  notes
      diagnostic messages.
"""
struct Analysis
    flat::Vector{_FlatMetric}
    case::Symbol
    const_idx::Vector{Int}
    groups::Vector{Vector{Int}}
    den_range::Vector{Tuple{Float64,Float64}}
    n_bilinear::Int
    remainder_const::Bool
    notes::Vector{String}
end

function Base.show(io::IO, a::Analysis)
    println(io, "Circularity analysis")
    println(io, "  case               : ", a.case)
    println(io, "  metrics            : ", length(a.flat))
    println(io, "  const. denominators: ",
            [a.flat[i].label for i in a.const_idx])
    println(io, "  L(x,y) constant    : ", a.remainder_const)
    println(io, "  variable-den groups: ")

    for (gidx, g) in enumerate(a.groups)
        println(io, "    group $gidx: ",
                [a.flat[i].label for i in g])
    end

    println(io, "  bilinear terms     : ", a.n_bilinear)

    for n in a.notes
        println(io, "  note: ", n)
    end
end

"""
    analyze(ocis, model; bounds = Dict(), use_lp = true, tol = 1e-6)

Classify the OCI defined by a vector of `SubIndex`. A denominator counts as
constant if it has no variables, or if its range over the LP relaxation of the
model collapses to a point (this detects denominators that the model's
equality constraints fix, e.g. inflow = demand).

Classification is done at the OCI level: metrics are flattened, split into a
constant-denominator set K0 (giving the affine remainder L(x,y)) and G groups of
variable-denominator metrics (grouped by identical denominator expression). The
case follows from G and K0 only:

  :A  G == 0                      -> OCI = L(x,y) is affine, no reformulation
  :B  G == 1 and K0 empty         -> aggregate numerators, cross-multiply (linear)
  :C  anything else               -> G >= 2, or G >= 1 together with K0 != empty;
                                     exact bilinear reformulation or McCormick
                                     relaxation (`strategy` in `add_circularity!`)

Metrics created with `normalized_indicator` have a constant denominator
(I_max - I_min) and therefore always belong to K0.

`bounds` overrides the LP range for a given metric label (e.g. "waste.diversion" =>
(1e-4, 0.05)). All metrics sharing a denominator should get the same bound, or leave
it to be computed for the group's representative metric. Call `analyze` BEFORE adding
the circularity constraint.
"""
function analyze(ocis::Vector{SubIndex}, model::Model;
                 bounds::Dict = Dict(), use_lp::Bool = true,
                 tol::Float64 = 1e-4)

    flat = flatten(ocis)
    notes = String[]

    # ---------------------------------------------------------------
    # Determine denominator ranges
    # ---------------------------------------------------------------
    ranges = Tuple{Float64,Float64}[]

    for m in flat
        if _is_const(m.den)
            c = m.den.constant
            push!(ranges, (c, c))

        elseif haskey(bounds, m.label)
            push!(ranges, Tuple{Float64,Float64}(bounds[m.label]))

        elseif use_lp
            push!(ranges, _range(model, m.den))

        else
            push!(ranges, (-Inf, Inf))
        end
    end

    # ---------------------------------------------------------------
    # Identify constant denominators
    # ---------------------------------------------------------------
    is_constden(r) =
        isfinite(r[1]) &&
        isfinite(r[2]) &&
        (r[2] - r[1] <= tol * max(1.0, abs(r[1])))

    const_idx = [
        i for i in eachindex(flat)
        if is_constden(ranges[i])
    ]

    var_idx = [
        i for i in eachindex(flat)
        if !(i in const_idx)
    ]

    # ---------------------------------------------------------------
    # Diagnostics on denominator positivity
    # ---------------------------------------------------------------
    for i in const_idx
        if !_is_const(flat[i].den)
            push!(
                notes,
                "Denominator of '$(flat[i].label)' has variables but " *
                "is constant on the feasible set; treated as constant " *
                "$(ranges[i][1])."
            )
        end

        ranges[i][1] > 0 ||
            push!(
                notes,
                "Denominator of '$(flat[i].label)' is <= 0."
            )
    end

    for i in var_idx
        lo, hi = ranges[i]

        if !(isfinite(lo) && isfinite(hi))
            push!(
                notes,
                "Denominator of '$(flat[i].label)' has infinite bounds; " *
                "provide `bounds` (required for Case C exact/McCormick " *
                "reformulations)."
            )

        elseif lo <= 0
            push!(
                notes,
                "Denominator of '$(flat[i].label)' can be <= 0 " *
                "(lower bound $lo). Case B/C requires D > 0. " *
                "Provide a positive lower bound."
            )
        end
    end

    # ---------------------------------------------------------------
    # Group variable-denominator metrics
    # ---------------------------------------------------------------
    groups = Vector{Vector{Int}}()

    for i in var_idx
        placed = false

        for g in groups
            rep = g[1]
            s = _prop_factor(flat[i].den, flat[rep].den)
            if s !== nothing
                if s != 1.0
                    # N/D_i = (N/s)/D_rep, since D_i = s * D_rep
                    _rescale!(flat, i, flat[rep].den, s)
                    ranges[i] = ranges[rep]
                    push!(notes, "Denominator of '$(flat[i].label)' is proportional " *
                                "(factor $s) to that of '$(flat[rep].label)'; " *
                                "numerator rescaled and indicators grouped.")
                end
                push!(g, i)
                placed = true
                break
            end
        end

        if !placed
            push!(groups, [i])
        end
    end

    G = length(groups)

    # All members of a group share the same denominator expression, so any
    # `bounds` override given for ONE of them must apply to all of them; the
    # rest of the code only ever reads `den_range[g[1]]`. Without this pass,
    # an override keyed to a non-representative member is silently discarded.
    for g in groups
        override = nothing
        for i in g
            if haskey(bounds, flat[i].label)
                b = Tuple{Float64,Float64}(bounds[flat[i].label])
                if override !== nothing && override != b
                    push!(
                        notes,
                        "Conflicting `bounds` given for group sharing the " *
                        "denominator of '$(flat[g[1]].label)'; using the " *
                        "bound from '$(flat[i].label)' = $b."
                    )
                end
                override = b
            end
        end
        if override !== nothing
            for i in g
                ranges[i] = override
            end
        end
    end

    # ---------------------------------------------------------------
    # L(x,y)
    #
    # True iff every numerator belonging to a constant-denominator
    # metric is itself constant.
    #
    # This is diagnostic information only. It no longer determines
    # whether the model is Case B or C.
    # ---------------------------------------------------------------
    remainder_const = all(_is_const(flat[i].num) for i in const_idx)

    # ---------------------------------------------------------------
    # NEW THREE-CASE CLASSIFICATION
    #
    # A: G = 0
    # B: G = 1 and K0 = empty
    # C: everything else
    # ---------------------------------------------------------------
    case =
        if G == 0
            :A
        elseif G == 1 && isempty(const_idx)
            :B
        else
            :C
        end

    # ---------------------------------------------------------------
    # Notes
    # ---------------------------------------------------------------
    if case == :A

        push!(
            notes,
            "All denominators are constant; OCI is affine."
        )

    elseif case == :B

        if length(groups[1]) > 1
            push!(
                notes,
                "All variable-denominator indicators share one common " *
                "denominator; their weighted numerators are aggregated " *
                "before cross-multiplication."
            )
        else
            push!(
                notes,
                "Single variable-denominator group; cross-multiplication " *
                "gives a linear circularity constraint."
            )
        end

    else

        if G >= 2
            push!(
                notes,
                "$G distinct variable denominators require " *
                "$G bilinear terms in the exact reformulation."
            )
        end

        if G >= 1 && !isempty(const_idx)
            push!(
                notes,
                "Variable-denominator indicators coexist with " *
                "constant-denominator indicators; L(x,y) is retained " *
                "explicitly in the Case C reformulation."
            )
        end

        if G == 0
            # This branch should never occur because G=0 is Case A.
            error("Internal classification error: Case C with G = 0.")
        end
    end

    return Analysis(
        flat,
        case,
        const_idx,
        groups,
        ranges,
        G,
        remainder_const,
        notes
    )
end

# ---------------------------------------------------------------------------
# 4. Generating the circularity constraint
# ---------------------------------------------------------------------------

struct CircularityHandle
    model::Model
    analysis::Analysis
    tau::Float64
    strategy::Symbol
    constraints::Vector{Any}
    vars::Vector{VariableRef}
    r::Vector{VariableRef}
end

"""
    _capture_values(model)

Store the current values of all model variables.
Used to transfer a McCormick candidate to a subsequent exact solve.
"""
function _capture_values(model::Model)
    vals = Dict{VariableRef, Float64}()

    for v in all_variables(model)
        if is_valid(model, v) && has_values(model)
            vals[v] = value(v)
        end
    end

    return vals
end


"""
    _set_initial_values!(model, vals)

Set the supplied variable values as initial points for the next solve.

These values are allowed to be infeasible for the new exact formulation.
"""
function _set_initial_values!(model::Model, vals::Dict{VariableRef, Float64})
    for (v, x) in vals
        if is_valid(model, v)
            set_start_value(v, x)
        end
    end

    return nothing
end

"""
    add_circularity!(
            ocis,
            model;
            tau,
            strategy = :exact,
            bounds = Dict(),
            normalized = true,
            analysis = nothing
        )

    Add `OCI >= tau` for the hierarchical circularity index.

   
    Case A:G = 0
        L(x,y) >= tau

    Case B: G = 1 and K₀ = ∅
        sum_k ω_k N_k(x,y) >= tau D(x,y)

    Case C covers all remaining structures:
        OCI = L(x,y) + sum_g N_g(x,y)/D_g(x,y) >= tau

    For `strategy = :exact`, one ratio variable r_g and denominator
    variable d_g are introduced for each distinct variable denominator:

        d_g = D_g(x,y)
        r_g d_g <= N_g(x,y)
        L(x,y) + sum_g r_g >= tau.

    The formulation is exact for positive denominators and nonnegative
    weights.

    For `strategy = :mccormick`, the bilinear relation is relaxed using
    McCormick inequalities. The resulting solution must be checked with
    `verify`, which recomputes the original OCI from the original ratios.
"""
function add_circularity!(
    ocis::Vector{SubIndex},
    model::Model;
    tau::Real,
    strategy::Symbol = :exact,
    bounds::Dict = Dict(),
    normalized::Bool = true,
    analysis::Union{Nothing,Analysis} = nothing
)

    strategy in (:exact, :mccormick) ||
        error("strategy must be :exact or :mccormick")

    A = analysis === nothing ?
        analyze(ocis, model; bounds = bounds) :
        analysis

    flat = A.flat

    cons = Any[]
    vars = VariableRef[]
    rs = VariableRef[]

    # ---------------------------------------------------------------
    # Construct L(x,y)
    #
    # L = sum_{k in K0} omega_k * N_k / c_k
    # ---------------------------------------------------------------
    const_expr = AffExpr(0.0)

    for i in A.const_idx
        c = A.den_range[i][1]

        c > 0 ||
            error(
                "Denominator of '$(flat[i].label)' is not positive " *
                "on the feasible set."
            )

        add_to_expression!(
            const_expr,
            flat[i].weight / c,
            flat[i].num
        )
    end
    # ===============================================================
    # CASE A
    # ===============================================================
    if A.case == :A

        push!(
            cons,
            @constraint(model, const_expr >= tau)
        )
    # ===============================================================
    # CASE B: CI = N_1 / D_1 >= tau
    # ===============================================================
    elseif A.case == :B

        g = A.groups[1]

        D = flat[g[1]].den

        lo, hi = A.den_range[g[1]]

        lo > 0 ||
            error(
                "Denominator must be > 0 for cross-multiplication; " *
                "provide a positive lower bound."
            )

        # `lo` is only a diagnostic bound on the LP relaxation; nothing so
        # far forces the MILP solution itself to respect it. Add the bound
        # explicitly so the cross-multiplied constraint below is only ever
        # evaluated at a well-defined (D > 0) point.
        push!(
            cons,
            @constraint(model, D >= lo)
        )

        Ntot = AffExpr(0.0)

        for i in g
            add_to_expression!(
                Ntot,
                flat[i].weight,
                flat[i].num
            )
        end

        push!(
            cons,
            @constraint(
                model,
                Ntot >= tau * D
            )
        )
    # ===============================================================
    # CASE C: OCI = L(x,y) + sum_g N_g / D_g >= tau
    #
    # Exact:
    #     r_g D_g <= N_g
    #
    # McCormick:
    #     r_g D_g <= N_g is relaxed using the McCormick envelope.
    # ===============================================================
    else

        G = length(A.groups)

        G >= 1 ||
            error("Internal error: Case C requires at least one variable denominator group.")

        for g in A.groups

            lo, hi = A.den_range[g[1]]

            (
                isfinite(lo) &&
                isfinite(hi) &&
                lo > 0
            ) || error(
                "Case C requires finite denominator bounds with " *
                "lower > 0 for group '$(flat[g[1]].label)'. " *
                "Pass `bounds`."
            )

            Dexpr = flat[g[1]].den

            # Aggregated weighted numerator
            Ng = AffExpr(0.0)

            # Total OCI weight represented by this group
            Wg = 0.0

            for i in g
                add_to_expression!(
                    Ng,
                    flat[i].weight,
                    flat[i].num
                )

                Wg += flat[i].weight
            end

            # Explicit denominator variable
            d = @variable(
                model,
                lower_bound = lo,
                upper_bound = hi
            )

            push!(vars, d)

            push!(
                cons,
                @constraint(model, d == Dexpr)
            )

            # Ratio contribution
            rU =
                normalized ?
                Wg :
                error(
                    "normalized = false is not implemented; " *
                    "pass normalized = true."
                )

            r = @variable(
                model,
                lower_bound = 0.0,
                upper_bound = rU
            )

            push!(vars, r)
            push!(rs, r)

            # -------------------------------------------------------
            # Exact bilinear formulation
            # -------------------------------------------------------
            if strategy == :exact

                push!(
                    cons,
                    @constraint(
                        model,
                        r * d <= Ng
                    )
                )

            # -------------------------------------------------------
            # McCormick relaxation
            # -------------------------------------------------------
            else

                # z = r*d
                #
                # Since z <= Ng and
                # z >= McCormick lower envelope,
                # the following two inequalities are sufficient
                # for the one-sided relaxation used here.
                push!(
                    cons,
                    @constraint(
                        model,
                        Ng >= lo * r
                    )
                )

                push!(
                    cons,
                    @constraint(
                        model,
                        Ng >= rU * d + hi * r - rU * hi
                    )
                )
            end
        end

        # -----------------------------------------------------------
        # Overall OCI constraint
        # -----------------------------------------------------------
        push!(
            cons,
            @constraint(
                model,
                const_expr + sum(rs) >= tau
            )
        )

        if strategy == :exact
            try
                set_optimizer_attribute(
                    model,
                    "NonConvex",
                    2
                )
            catch
                @info(
                    "Could not set NonConvex=2; make sure the " *
                    "selected solver supports nonconvex bilinear constraints."
                )
            end
        end
    end

    return CircularityHandle(
        model,
        A,
        Float64(tau),
        strategy,
        cons,
        vars,
        rs
    )
end

"""Convenience overload for the reversed argument order (`model` first); forwards to
the canonical `add_circularity!(ocis, model; ...)` method."""
function add_circularity!(model::Model, ocis::Vector{SubIndex}; kwargs...)
    return add_circularity!(ocis, model; kwargs...)
end

"""Remove the constraint and auxiliary variables added by `add_circularity!` (for tau sweeps)."""
function remove_circularity!(h::CircularityHandle)
    for c in h.constraints
        delete(h.model, c)
    end
    for v in h.vars
        delete(h.model, v)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 5. Verification 
# ---------------------------------------------------------------------------

struct CircularityReport
    case
    strategy
    tau

    # Exact and relaxed circularity
    oci_exact
    oci_relaxed
    violation

    # Detailed results
    ratios
    subindex_values
    r_overstatement

    # Feasibility
    normalization_ok
    ok

    # Objective / optimality information
    objective
    lower_bound
    upper_bound
    abs_gap
    rel_gap
    gap_ok
end

function Base.show(io::IO, r::CircularityReport)
    println(io, "Circularity verification")
    println(io, "  Case / Strategy : ", r.case, " / ", r.strategy)
    println(io, "  tau             : ", r.tau)
    println(io, "  OCI (exact)     : ", round(r.oci_exact, digits = 6))
    println(io, "  OCI (relaxed)   : ", round(r.oci_relaxed, digits = 6))
    println(io, "  Violation       : ", round(r.violation, digits = 6))
    println(io, "  r overstatement : ", round(r.r_overstatement, digits = 6))
    println(io, "  ratios in [0,1] : ", r.normalization_ok)
    println(io, "  VALID           : ", r.ok)
    println(io, "  Sub-indices:")
    for (k, v) in r.subindex_values
        println(io, "    ", k, " = ", round(v, digits = 6))
    end
    println(io, "  metrics:")
    for (k, v) in r.ratios
        println(io, "    ", k, " = ", round(v, digits = 6))
    end
end

"""
Verify the solution against the ORIGINAL circularity formulation.

- Recomputes every ratio from the original numerator/denominator variables.
- Never uses auxiliary ratio variables to determine exact feasibility.
- Recomputes each SubIndex from the exact ratios.
- Reports the OCI seen by the relaxed formulation (`oci_relaxed`).
- For Case C, reports the maximum overstatement of the auxiliary ratio variables.
- If `lower_bound` is supplied, computes an upper bound from the current
  original-feasible solution and the corresponding optimality gap.
"""
function verify(
    h::CircularityHandle;
    tol::Float64 = 1e-4,
    lower_bound::Union{Nothing,Float64} = nothing,
    gap_tol::Float64 = 1e-4
)

    flat = h.analysis.flat

    # ============================================================
    # 1. EXACT RATIOS FROM ORIGINAL VARIABLES
    # ============================================================

    ratios = Dict{String,Float64}()
    vals = Float64[]

    for m in flat

        n = value(m.num)
        d = value(m.den)

        d > 0 || error(
            "Denominator of '$(m.label)' is not positive at the solution."
        )

        ratio = n / d

        push!(vals, ratio)
        ratios[m.label] = ratio
    end


    # ============================================================
    # 2. EXACT OCI: recomputed from N(x,y)/D(x,y).
    # ============================================================

    oci_exact = sum(flat[i].weight * vals[i] for i in eachindex(flat))
    violation = max(0.0, h.tau - oci_exact)

    # ============================================================
    # 3. OCI RELAXED MODEL
    # ============================================================

    oci_relaxed = if h.analysis.case == :C

        Lval = 0.0

        for i in h.analysis.const_idx

            c = h.analysis.den_range[i][1]

            Lval += flat[i].weight * value(flat[i].num) / c
        end

        Lval + sum(value(rg) for rg in h.r; init = 0.0)
    else
        oci_exact
    end

    # ============================================================
    # 4. EXACT SUBINDEX VALUES
    # ============================================================

    subindex_values = Dict{String,Float64}()
    subindex_weight_sum = Dict{String,Float64}()

    for m in flat
        subindex_weight_sum[m.subindex] = get(subindex_weight_sum, m.subindex, 0.0) + m.weight
    end

    for (i, m) in enumerate(flat)

        w_local =
            m.weight /
            subindex_weight_sum[m.subindex]

        subindex_values[m.subindex] =
            get(
                subindex_values,
                m.subindex,
                0.0
            ) + w_local * vals[i]
    end

    # ============================================================
    # 5. AUXILIARY RATIO OVERSTATEMENT
    # ============================================================

    r_overstatement = 0.0

    if h.analysis.case == :C

        for (k, g) in enumerate(h.analysis.groups)

            exact_g =sum(flat[i].weight * vals[i] for i in g)

            auxiliary_g = value(h.r[k])

            r_overstatement =max(r_overstatement, auxiliary_g - exact_g)
        end
    end

    # ============================================================
    # 6. BASIC ORIGINAL-PROBLEM FEASIBILITY
    # ============================================================
    normalization_violation = maximum(max(0.0, -r, r - 1.0) for r in vals)

    normalization_ok = normalization_violation <= tol

    ok = violation <= tol && normalization_ok
    
    # ============================================================
    # 7. OBJECTIVE VALUE
    # ============================================================

    objective =
        try
            objective_value(h.model)
        catch
            NaN
        end


    # ============================================================
    # 8. LOWER / UPPER BOUNDS
    # ============================================================

    lb = lower_bound

    ub =
        ok && isfinite(objective) ?
        objective :
        nothing

    # ============================================================
    # 9. OPTIMALITY GAP
    # ============================================================
    abs_gap = nothing
    rel_gap = nothing
    gap_ok = nothing

    if lb !== nothing && ub !== nothing

        abs_gap = ub - lb

        # Numerical protection
        if abs_gap < 0 && abs_gap > -tol
            abs_gap = 0.0
        end

        rel_gap = abs_gap / max(1.0, abs(ub))

        gap_ok = rel_gap <= gap_tol
    end

    # ============================================================
    # 10. RETURN REPORT
    # ============================================================

    return CircularityReport(
        h.analysis.case,
        h.strategy,
        h.tau,

        oci_exact,
        oci_relaxed,
        violation,

        ratios,
        subindex_values,
        r_overstatement,

        normalization_ok,
        ok,

        objective,
        lb,
        ub,
        abs_gap,
        rel_gap,
        gap_ok
    )
end


"""
    optimize_circularity!(model, ocis;
                          tau,
                          strategy = :exact,
                          bounds = Dict(),
                          normalized = true,
                          analysis = nothing,
                          validate = true,
                          refine = false,
                          tol = 1e-6)

Solve a circularity-constrained optimization problem.

For `strategy = :mccormick`:

1. Solve the McCormick relaxation.
2. Validate the resulting solution against the original fractional OCI.
3. If the original OCI constraint is violated and `refine = true`,
   rebuild the circularity constraint using the exact formulation and
   use the McCormick solution as an initial point.

The McCormick solution is therefore treated as a candidate/initial point,
not as a guaranteed feasible solution of the original fractional problem.
"""
function optimize_circularity!(
    model::Model,
    ocis::Vector{SubIndex};
    tau::Real,
    strategy::Symbol = :exact,
    bounds::Dict = Dict(),
    normalized::Bool = true,
    analysis::Union{Nothing,Analysis} = nothing,
    validate::Bool = true,
    refine::Bool = false,
    tol::Float64 = 1e-4
)

    A = analysis === nothing ?
        analyze(ocis, model; bounds = bounds) :
        analysis

    # ------------------------------------------------------------
    # 1. First solve
    # ------------------------------------------------------------
    h = add_circularity!(
        model,
        ocis;
        tau = tau,
        strategy = strategy,
        bounds = bounds,
        normalized = normalized,
        analysis = A
    )

    t0 = time()
    optimize!(model)
    time_initial = time() - t0

    status_first = termination_status(model)

    # No solution
    if !has_values(model)
        return (
            status = status_first,
            first_status = status_first,
            strategy = strategy,
            refined = false,
            handle = h,
            report = nothing,
            time_initial = time_initial,
            time_refinement = 0.0,
            time_total = time_initial
        )
    end

    # ------------------------------------------------------------
    # 2. Validate McCormick candidate
    # ------------------------------------------------------------
    report = validate ? verify(h; tol = tol) : nothing

    # If exact strategy, nothing to refine
    if strategy == :exact
        return (
            status = status_first,
            first_status = status_first,
            strategy = strategy,
            refined = false,
            handle = h,
            report = report,
            time_initial = time_initial,
            time_refinement = 0.0,
            time_total = time_initial
        )
    end

    # ------------------------------------------------------------
    # 3. If McCormick candidate is valid, return it
    # ------------------------------------------------------------
    if !refine || report.ok
        return (
            status = status_first,
            first_status = status_first,
            strategy = strategy,
            refined = false,
            handle = h,
            report = report,
            time_initial = time_initial,
            time_refinement = 0.0,
            time_total = time_initial
        )
    end

    # ------------------------------------------------------------
    # 4. Candidate violates original fractional constraint
    # ------------------------------------------------------------
    @info "McCormick candidate violates the original circularity constraint." *
          " Refining with the exact formulation."

    # Save candidate values BEFORE deleting the McCormick auxiliaries.
    candidate = _capture_values(model)

    # Remove McCormick constraint and auxiliary variables.
    remove_circularity!(h)

    # ------------------------------------------------------------
    # 5. Add exact formulation
    # ------------------------------------------------------------
    h_exact = add_circularity!(
        model,
        ocis;
        tau = tau,
        strategy = :exact,
        bounds = bounds,
        normalized = normalized,
        analysis = A
    )

    # ------------------------------------------------------------
    # 6. Use McCormick candidate as initial point
    # ------------------------------------------------------------
    _set_initial_values!(model, candidate)

    # ------------------------------------------------------------
    # 7. Solve exact problem
    # ------------------------------------------------------------
    t1 = time()
    optimize!(model)
    time_refinement = time() - t1

    status_exact = termination_status(model)

    time_total = time_initial + time_refinement

    # No exact solution
    if !has_values(model)
        return (
            status = status_exact,
            first_status = status_first,
            strategy = strategy,
            refined = true,
            handle = h_exact,
            report = nothing,
            time_initial = time_initial,
            time_refinement = time_refinement,
            time_total = time_total
        )
    end

    # ------------------------------------------------------------
    # 8. Verify exact solution
    # ------------------------------------------------------------
    report_exact = verify(h_exact; tol = tol)

    return (
        status = status_exact,
        first_status = status_first,
        strategy = strategy,
        refined = true,
        handle = h_exact,
        report = report_exact,
        time_initial = time_initial,
        time_refinement = time_refinement,
        time_total = time_total
    )
end

# ---------------------------------------------------------------------------
# 6. Tau sweep (epsilon-constraint Pareto front)
# ---------------------------------------------------------------------------

"""
    sweep_tau(ocis, model, taus; strategy = :exact, bounds = Dict(), normalized = true)

For each tau: add the constraint, optimize, verify, remove. Returns a vector of
NamedTuples (tau, status, objective, report). Analysis is done once.
"""
function sweep_tau(ocis::Vector{SubIndex}, model::Model, taus;
                   strategy::Symbol = :exact, bounds::Dict = Dict(),
                   normalized::Bool = true)
    A = analyze(ocis, model; bounds = bounds)
    out = NamedTuple[]
    for t in taus
        h = add_circularity!(ocis, model; tau = t, strategy = strategy,
                             normalized = normalized, analysis = A)
        optimize!(model)
        st = termination_status(model)
        if has_values(model)
            push!(out, (tau = t, status = st, objective = objective_value(model), report = verify(h)))
        else
            push!(out, (tau = t, status = st, objective = NaN, report = nothing))
        end
        remove_circularity!(h)
    end
    return out
end

end # module
