module ZeroFindingMethods

export bracket, quasistep

MINRED = 0.25
REDFAC = 0.5
MINF3RED = 0.5

struct TraceObject
    records::Vector{Any}
    TraceObject() = new(Vector{Any}(undef, 0))
end

push(tr::TraceObject, v::Any) = push!(tr.records, v)

function bracket(f, br::NTuple{2}, tr=TraceObject())
    x0, x1 = br
    f0, f1 = f(x0), f(x1)
    f0 * f1 > 0 && throw(ArgumentError("no bracketing staring values"))

    nextmidpoint = true
    fcount = 2
    while x1 - x0 > 0 && f0 != 0 && f1 != 0
        x2 = nextmidpoint ? middle(x0, x1) : secant(x0, x1, f0, f1)
        if x2 == x0 || x2 == x1
            break
        end
        f2 = f(x2)
        fcount += 1
        push(tr, (;x0, x2, x1, f0, f2, f1, fcount, nextmidpoint, status=1))
        println("entering loop: $(nextmidpoint ? "BI" : "SE") x0 = $x0, x2 = $x2, x1 = $x1, f0 = $f0, f2 = $f2, f1 = $f1")

        minred = abs(x1 - x0) * MINRED # Constant minred in (0, 0.5)
        if samesign(f2, f0) && abs(x2 - x0) < minred
            f0 *= REDFAC
        elseif samesign(f2, f1) && abs(x2 - x1) < minred
            f2 *= REDFAC
        end

        x3 = quasistep(x0, x2, x1, f0, f2, f1)

        if isfinite(x3)
            nextmidpoint = false
        end
        if isfinite(x3) && false
            f3 = f(x3)
            fcount += 1

            push(tr, (;x0, x2, x3, x1, f0, f2, f3, f1, fcount, nextmidpoint, status=2))
            println("found x3 = $x3 f3 = $f3")
            if samesign(f2, f3)
                if !nextmidpoint
                    println("Newton step function sign not as expected")
                    nextmidpoint = true
                elseif abs(f3) < abs(f2) * MINF3RED
                    nextmidpoint = false
                end
                x2, f2 = x3, f3
            else
                if x2 < x3
                    x0, x1, f0, f1 = x2, x3, f2, f3
                else
                    x0, x1, f0, f1 = x3, x2, f3, f2
                end
                nextmidpoint = false
                continue
            end
        else
            nextmidpoint = true
        end
        if samesign(f0, f2)
            f0 = f2
            x0 = x2
        else
            f1 = f2
            x1 = x2
        end
    end
    if f0 == 0
        x1 = x0
    end
    if f1 == 0
        x0 = x1
    end
    push(tr, (;x0, x1, f0, f1, fcount, status = 3))
    return tr
end

"""
    quasistep(x0, xm, x1, f0, fm, f1)

Input condition
x0 < xm < x1 and f0 * f1 < 0

Return NaN if there is no good quasi-newton step
otherwise return x3 between x0 and x1
"""
function quasistep(x0, x2, x1, f0, f2, f1)

    tr(t, x0, x1) = (t * (x1 - x0) + x0 + x1) / 2
    INVALID = NaN
    #
    if f0 > 0
        f0, f1 = f1, f0
    end
    if f2 < f0 || f2 > f1
        println("Newton not tried because f2 = $f2 not in ($f0, $f1)")
        return INVALID
    end

    # transform to interval [-1, 1]
    t2 = ((x2 - x0) + (x2 - x1)) / (x1 - x0)

    if !(-1 < t2 < 1)
        x3 = middle(x0, x1)
        println("Newton can't be calculated because accuracy limit reached")
        return x3
    end

    # construct interpolation parabola P through (-1,f0), (x2, f2), (1,f1)
    fm = middle(f0, f1)
    h = f1 / 2 - f0 / 2 # slope of secant
    xs = -fm / h # root of secant function
    g = 2 * ((t2 - xs) * h - f2) / (1 - t2^2) # P(x) = fm + h * x - 1/2 g * (1-x^2)

    a = (x0 + x1) / 2
    b = (x1 - x0) / 2
    hh = h / b
    gg = g / b^2
    u, v, w = fm - hh * a - gg / 2 * (b^2 - a^2), hh - g / b^2 * a, gg / 2

    #println("h = $h, g = $g in P(t) = $fm + h * t - 1/2*g*(1-t^2) =^= $u + $v * x + $w * x^2")

    println("H = $(h/(x1-x0)) G = $(g / (x1-x0)^2) Δx = $(x1-x0)")

    # check P'(-1) > 0 and P'(1) > 0 - equivalent to for all x in [-1,1]; P(x) = h + g * x
    PARAM = 0.2
    if abs(g) > h * (1.0 - PARAM)
        println("Newton rejected because g = $g and abs(g) >= $h * (1 - $PARAM)")
        return INVALID
    end
    h2 = h + g * t2 # P'(t2)

    # calculate Newton step for P at x2: t3 = t2 - f2 / P'(t2)
    t3 = t2 - f2 / h2

    # check if Newton step is inside interval
    if !(-1 <= t3 <= 1)
        println("Newton rejected because x3 = $(tr(t3,x0,x1)) not in [$x0, $x1]")
        return INVALID
    end

    println("accepted Newton x3 = $(tr(t3,x0,x1)) in [$x0, $x1]")

    # transform back to interval [x0, x1]

    x3 = tr(t3, x0, x1)
    return x3
end

middle(a, b) = a / 2 + b / 2
samesign(a, b) = a > 0 && b > 0 || a < 0 && b < 0
secant(x0, x1, f0, f1) = x0 - (f0 / (f1 - f0)) * (x1 - x0)

end
