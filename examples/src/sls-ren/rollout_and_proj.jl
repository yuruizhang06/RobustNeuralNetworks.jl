using JuMP
using MosekTools
using RobustNeuralNetworks

includet("./utils.jl")

function rollout(G::lti, Q::ContractingRENParams, w)
    Qe = REN(Q)
    nx, nu = G.nx, G.nu
    batch = size(w[1], 2)
    X1 = (zeros(Qe.nx, batch))

    function f(X_1, t)
        h_1 = X_1 
        ht, ψt = Qe(h_1, w[t])
        # xt = G(x_1, u_1, w[t])
        # ht, v = Qe(h_1, w_1) 
        # wht = xt - v[1:nx,:]
        # hnt, vt= Qe(ht, wht) 
        # ut = vt[nx+1:nx+nu, :]
         ψx = ψt[1:nx,:]
         ψu = ψt[nx+1:nx+nu,:]
        # validation
        Xt = ht
        zt  = vcat(ψx , ψu)
        return Xt, zt
    end

    md = Flux.Recur(f,X1)
    z = md.(1:length(w))

    return z
end

function validation(G::lti, Q::ContractingRENParams, w)
    Qe = REN(Q)
    nx, nu = G.nx, G.nu
    batch = size(w[1], 2)
    X1 = (zeros(nx, batch), zeros(Qe.nx, batch), zeros(nx, batch),
         zeros(nu, batch))
    ψxs = zeros(1,batch)
    ψus = zeros(1,batch)

    function f(X_1, t)
        x_1, h_1, w_1, u_1 = X_1 
        xt = G(x_1, u_1, w[t])
        ht, v = Qe(h_1, w_1) 
        wht = xt - v[1:nx,:]
        hnt, vt= Qe(ht, wht) 
        ut = vt[nx+1:nx+nu, :]

        # validation
        Xt = (xt, ht, wht, ut)
        zt  = vcat(xt, ut)

        return Xt, zt
    end

    md = Flux.Recur(f,X1)
    z = md.(1:length(w))
 
    for i in 1:length(w)
        ψxs = vcat(ψxs, z[i][1:nx,:])
        ψus = vcat(ψus, z[i][nx+1:nx+nu,:])
    end

    return z, ψxs, ψus
end

function proj!(G::lti, Q::ContractingRENParams)
    Qe=REN(Q)
    nx, nu = G.nx, G.nu 
    nqx, nqv = Qe.nx, Qe.nv 

    prob = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    #set variable 
    @variable(prob,  Cx[1:nx,1:nqx])
    @variable(prob,  Cu[1:nu,1:nqx])
    @variable(prob, Du1[1:nu,1:nqv])
    @variable(prob, Du2[1:nu,1:nx])
    @variable(prob, bx[1:nx])
    @variable(prob, bu[1:nu])

    # equality constraint
    @constraint(prob, Cx*Qe.explicit.A  .== G.A*Cx + G.B*Cu )
    @constraint(prob, Cx*Qe.explicit.B1 .== G.B*Du1)
    @constraint(prob, Cx*Qe.explicit.B2 .== G.B*Du2 + G.A)
    @constraint(prob, Cx*Qe.explicit.bx+(I-G.A)*bx .== G.B*bu')

    D21 = [zeros(nx, nqv); Du1]
    D22 = [I;  Du2]
    by = [bx; bu]

    obj = sum((Cx-Qe.explicit.C2[1:nx,:]).^2) + sum((Cu-Qe.explicit.C2[nx+1:end,:]).^2)
        + sum((D21-Qe.explicit.D21).^2) + sum((D22-Qe.explicit.D22).^2)
        + sum((by-Qe.explicit.by).^2)
    @objective(prob, Min, obj)
    optimize!(prob)

    Q.direct.C2 = [value.(Cx); value.(Cu)]
    Q.direct.D21 = [zeros(nx, nqv); value.(Du1)]
    Q.direct.D22 = [zeros(nx, nx);  value.(Du2)]
    Q.direct.by = value.(by)

    𝕘 = vcat(vec(value.(Cx)),vec(value.(Cu)),vec(value.(Du1)),
        vec(value.((Du2))),value.(by))
end

function rollout(G::lti, Q::SystemlevelRENParams, w)
    Qe = REN(Q)
    nx, nu = G.nx, G.nu
    batch = size(w[1], 2)
    X1 = (zeros(Qe.nx, batch))

    function f(X_1, t)
        h_1 = X_1 
        ht, ψt = Qe(h_1, w[t])
        # xt = G(x_1, u_1, w[t])
        # ht, v = Qe(h_1, w_1) 
        # wht = xt - v[1:nx,:]
        # hnt, vt= Qe(ht, wht) 
        # ut = vt[nx+1:nx+nu, :]
         ψx = ψt[1:nx,:]
         ψu = ψt[nx+1:nx+nu,:]
        # validation
        Xt = ht
        zt  = vcat(ψx , ψu)
        return Xt, zt
    end

    md = Flux.Recur(f,X1)
    z = md.(1:length(w))

    return z
end

function validation(G::lti, Q::SystemlevelRENParams, w)
    Qe = REN(Q)
    nx, nu = G.nx, G.nu
    batch = size(w[1], 2)
    X1 = (zeros(nx, batch), zeros(Qe.nx, batch), zeros(nx, batch),
         zeros(nu, batch))
    ψxs = zeros(1,batch)
    ψus = zeros(1,batch)

    function f(X_1, t)
        x_1, h_1, w_1, u_1 = X_1 
        xt = G(x_1, u_1, w[t])
        ht, v = Qe(h_1, w_1) 
        wht = xt - v[1:nx,:]
        hnt, vt= Qe(ht, wht) 
        ut = vt[nx+1:nx+nu, :]

        # validation
        Xt = (xt, ht, wht, ut)
        zt  = vcat(xt, ut)

        return Xt, zt
    end

    md = Flux.Recur(f,X1)
    z = md.(1:length(w))
 
    for i in 1:length(w)
        ψxs = vcat(ψxs, z[i][1:nx,:])
        ψus = vcat(ψus, z[i][nx+1:nx+nu,:])
    end

    return z, ψxs, ψus
end