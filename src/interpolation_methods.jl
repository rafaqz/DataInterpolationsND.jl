Base.@propagate_inbounds function _interpolate!(
        out::Union{Number, AbstractArray},
        A::NDInterpolation{N},
        ts::Tuple{Vararg{Any, N}},
        idx::Tuple{Vararg{Any, N}},
        derivative_orders::Tuple{Vararg{Any, N}},
        multi_point_index
) where {N}
    (; interp_dims, cache, u) = A

    out,
    valid_derivative_orders = check_derivative_order(
        interp_dims, derivative_orders, ts, out)
    valid_derivative_orders || return out # Array was zeroed out in this case
    if isnothing(multi_point_index)
        multi_point_index = map(_ -> nothing, interp_dims)
    end

    # Setup
    coeffs = map(coefficients, interp_dims, derivative_orders, multi_point_index, ts, idx)

    return apply_stencils!(out, u, interp_dims, cache, coeffs, ts, idx) 
end

@generated function apply_stencils!(out, u, interp_dims, cache, coeffs, ts, idx) 
    # Get stencils for each dimension
    stencils = map(stencil, interp_dims.types)

    # Initialise 
    setups = Expr(:block)
    mul_exprs = Expr[]

    # Define operators for if this a broadcast or assignment to a variable
    eq, mul, add, div = out <: AbstractArray ? (:.=, :.*, :.+, :./) : (:(=), :*, :+, :/)

    # Unroll stencil product setup
    for I in Iterators.product(stencils...)
        idx_str = join(map(string, I))
        index_I = Symbol(:index, idx_str)
        product_I = Symbol(:product, idx_str)
        setup = quote
            $index_I = map(index, interp_dims, ts, idx, $I)
            c = map(getindex, coeffs, $I)
            $product_I = prod(c)
            if cache isa NURBSWeights
                K = removeat(NoInterpolationDimension, $index_I, interp_dims)
                $product_I *= cache.weights[K...]
                denom += $product_I
            end
        end
        push!(setups.args, setup)
        u_expr = out <: AbstractArray ? :(view(u, $index_I...)) : :(u[$index_I...])
        push!(mul_exprs, Expr(:call, mul, product_I, ))
    end

    # Define sum expression around all mul operations
    sum_expr = mul_exprs[1]
    for m in mul_exprs[2:end]
        sum_expr = Expr(:call, add, sum_expr, m)
    end

    # Divide by denom when NURBSWeights for final rhs expression
    rhs_expr = cache <: NURBSWeights ? Expr(:call, div, sum_expr, :denom) : sum_expr

    # Write out in place or as an assingment
    assignment_expr = Expr(eq, :out, rhs_expr)

    # Combine into the full function
    quote 
        # assignment_expr = $(QuoteNode(assignment_expr))
        # @show assignment_expr
        # dump(assignment_expr)
        # @show typeof(out)
        denom = zero(eltype(out))
        $setups
        $assignment_expr

        return out
    end
end

function check_derivative_order(dims::Tuple, derivative_orders::Tuple, ts::Tuple, out)
    itr = map(tuple, dims, derivative_orders, ts)
    # Fold over itr for all dims, combining out and valid
    foldl(itr; init = (out, true)) do (acc_out, acc_valid), (d, d_o, t)
        dim_out, dim_valid = check_derivative_order(d, d_o, t, acc_out)
        dim_out, dim_valid & acc_valid
    end
end
check_derivative_order(::AbstractInterpolationDimension, d_o, t, out) = (out, true)
check_derivative_order(::LinearInterpolationDimension, d_o, t, out) = (out, d_o <= 1)
function check_derivative_order(d::ConstantInterpolationDimension, d_o, t, out)
    if d_o > 0
        # Check if t is on the boundary between constant steps and if so return nans
        return if isempty(searchsorted(d.t, t))
            (out, false)
        else
            (typed_nan(out), false)
        end
    else
        (out, true)
    end
end

stencil(::T) where T = stencil(T)
stencil(::Type{<:LinearInterpolationDimension}) = (1, 2)
stencil(::Type{<:ConstantInterpolationDimension}) = 1
stencil(::Type{<:NoInterpolationDimension}) = 1
stencil(::Type{<:BSplineInterpolationDimension{Degree}}) where Degree = 
    ntuple(identity, Val{Degree + 1}())

struct One <: Real end
Base.:(*)(::One, x::Number) = x
Base.:(*)(x::Number, ::One) = x

# Precalculate coefficient/s
function coefficients(
        d::LinearInterpolationDimension, derivative_order, multi_point_index, t, i)
    @inbounds t₁ = d.t[i]
    @inbounds t₂ = d.t[i + 1]
    t_vol_inv = inv(t₂ - t₁)
    a = (iszero(derivative_order) ? t₂ - t : -one(t)) * t_vol_inv
    b = (iszero(derivative_order) ? t - t₁ : one(t)) * t_vol_inv
    return (a, b)
end
function coefficients(
        ::ConstantInterpolationDimension, derivative_order, multi_point_index, t, i)
    true
end
coefficients(::NoInterpolationDimension, derivative_order, multi_point_index, t, i) = One()
function coefficients(
        d::BSplineInterpolationDimension, derivative_order, multi_point_index, t, i)
    get_basis_function_values(d, t, i, derivative_order, multi_point_index)
end

index(::LinearInterpolationDimension, t, idx, i) = idx + i - 1
# TODO: this should happen outside of the loop
index(d::ConstantInterpolationDimension, t, idx, i) = t >= d.t[end] ? length(d.t) : idx[i]
index(::NoInterpolationDimension, t, idx, i) = idx
index(d::BSplineInterpolationDimension, t, idx, i) = idx + i - degree(d) - 1
