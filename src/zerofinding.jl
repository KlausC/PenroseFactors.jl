module ZeroFindingMethods

export bracket, updateg

struct TraceObject
    records::Vector{Any}
    TraceObject() = new(Vector{Any}(undef, 0))
end

push(tr::TraceObject, v::Any) = push!(tr.records, v)

function bracket(f, br::NTuple{2}, g=0.0; tr=TraceObject())
    x0, x1 = br
    f0, f1 = f(x0), f(x1)
    f0 * f1 > 0 && throw(ArgumentError("no bracketing staring values"))

    # initial guess for second derivative of f
    #g = oftype(f0, NaN)
    fcount = 2
    hprev = oftype(x1 - x0, Inf)
    S = false
    while x1 - x0 > 0 && f0 != 0 && f1 != 0
        print("entering loop: x0 = $x0, x1 = $x1, f0 = $f0, f1 = $f1, g = $g")
        h = x1 - x0
        S =  S && (h <= 0.51 * hprev)
        s = (f1 - f0) / h
        xm = middle(x0, x1)
        xs = secant(x0, x1, f0, f1)
        A = f0 > 0
        B = abs(f0) > abs(f1)
        C = g > 0
        next = S ? ((A === (B === C)) ? :n : :s) : :m
        if !isfinite(g) || next === :m
            x2 = xm
        elseif next === :s
            x2 = xs
        else # next === :n
            x3 = x0 - f0 / (s - g * h / 2)
            x4 = x1 - f1 / (s + g * h / 2)
            x2 = if x3 <= x0 && x4 >= x1
                middle(x0, x1)
            else
                x3 <= x0 ? x4 : x4 >= x1 ? x3 : A ? min(x3, x4) : max(x3, x4)
            end
        end
        if x2 == x0 || x2 == x1
            break
        end
        f2 = f(x2)
        fcount += 1
        push(tr, (; next, x0, x2, x1, f0, f2, f1, fcount, g, status=1))

        g = updateg(g, x0, x2, x1, f0, f2, f1)

        if samesign(f0, f2)
            f0 = f2
            x0 = x2
        else
            f1 = f2
            x1 = x2
        end
        S = (f2 > 0) === (A === B)
        hprev = h
        println()
    end
    if f0 == 0
        x1 = x0
    end
    if f1 == 0
        x0 = x1
    end
    push(tr, (; next=:e, x0, x1, f0, f1, fcount, g, status=3))
    return tr
end

function updateg(g, x0, x2, x1, f0, f2, f1)
    alpha = 1.0
    h = x1 - x0
    b = (f1 - f0) / h
    h = h / 2
    h2 = x2 - middle(x0, x1)
    h2h = h^2 - h2^2
    if h2 <= 0
        c = ((f2 - f0) / (x2 - x0) - b) / (x2 - x1) * 2
    else
        c = ((f1 - f2) / (x1 - x2) - b) / (x2 - x0) * 2
    end
    d = 200abs(g) # eps(max(abs(f2), abs(f0))) / h2h * 2
    g = isfinite(g) ? clamp(c, g - d, g + d) * alpha + g * (1.0 - alpha) : c
    return g
end

middle(a, b) = a / 2 + b / 2
secant(x0, x1, f0, f1) = x0 - (f0 / (f1 - f0)) * (x1 - x0)
samesign(a, b) = a > 0 && b > 0 || a < 0 && b < 0
clampinterval(x, a, b) = clamp(x, min(a, b), max(a, b))

end
