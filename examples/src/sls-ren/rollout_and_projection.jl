using JuMP
using MosekTools
using Ipopt
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
        # validation
        Xt = ht
        zt  = ψt
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
        # wht = xt - v[1:nx,:]
        wht = (xt - Qe.explicit.C2[1:nx,:]*ht .- Qe.explicit.by[1:nx,:])*0.5
        hnt, vt= Qe(ht, wht) 
        
        # stop_here()
        ψx = vt[1:nx,:]
        ut = vt[nx+1:nx+nu, :]

        # validation
        Xt = (ψx, ht, wht, ut)
        # zt  = vcat(xt, ut)
        zt = vt
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

function solve_lqr(G::lti, L, step, x0, ubar)
    Q = diagm(0 => L[1:G.nx]) 
    R = diagm(0 => L[G.nx+1:G.nx+G.nu])
    nx, nu = G.nx, G.nu 
    # X, E, K, Z = ared(G.A, G.B, R, Q, zeros(G.nx, G.nu))
    prob = Model(Ipopt.Optimizer)
    #set variable 
    @variable(prob,  x[1:nx,1:step])
    @variable(prob,  u[1:nu,1:step-1])

    # equality constraint
    # @constraint(prob, x.== G.A*Cx + G.B*Cu )
    @constraint(prob, x[:, 2:end] .== G.A * x[:, 1:end-1] + G.B * u[:, 1:end])
    @constraint(prob, x[:, 1] .== x0)

    # obj = sum(x[:,i]' * Q * x[:,i] + u[:,i]' * R * u[:,i] for i =1:step-1)
    obj = sum(x[:,i]' * Q * x[:,i] + u[:,i]' * R * u[:,i] + max(u[:,i].-ubar, 5) for i =1:step-1)

    @objective(prob, Min, obj)
    optimize!(prob)

    u_out = value.(u)
    return u_out
end

function rollout(G::lti, Q::SystemlevelRENParams, w)
    Qe = REN(Q)
    nx, nu = G.nx, G.nu
    batch = size(w[1], 2)
    ψxs = zeros(1, batch)
    ψus = zeros(1, batch)

    # h_1 = zeros(Qe.nx, batch)
    # w_1 = zeros(nx, batch)
    # h0, ψ_1 = Qe(h_1, w_1)
    x1 = zeros(nx, batch)
    h_1 = zeros(Qe.nx, batch)
    w_1 = zeros(nx, batch)
    hr0, ψr_1 = Qe(h_1, w_1)
    u_1 = zeros(nu, batch)
    
    # X0 = G(x1, u_1, w_1)
    # wh0 = x1 - Qe.explicit.C2[1:size(A,1),:]*hr0 .- Qe.explicit.by[1:size(A,1),:]
    wh0 = x1 - Qe.explicit.C2[1:nx,:]*hr0 .- Qe.explicit.by[1:nx,:]
    hr1, ψr0 = Qe(hr0, wh0)
    ψur0 = ψr0[nx+1:end,:]
    ψxr0 = ψr0[1:nx,:]

    # h1, ψ0 = ren(h0, wh0)

    X1 = (x1, hr1, ψur0)

    function f(X_1, t)
        x_1, ht, u_1 = X_1 #system state, hidden state, ̂w, control input
        xt = G(x_1, u_1, w[t]) # system state at time t
        hnt, ψt = Qe(ht, w[t])
        x_t = ψt[1:nx,:] # ψ_x at time t
        ut = ψt[nx+1:end, :] # ψ_u at time t
        # ut_ = tanh.(0.01*ut)
        # println(norm(xt-x_t))
        Xt = (xt, hnt, ut)
        # return hnt, ψt
        return Xt, ψt
    end

    md = Flux.Recur(f,X1)
    z = md.(1:length(w))

    for i in 1:length(w)
        ψxs = vcat(ψxs, z[i][1:nx,:])
        ψus = vcat(ψus, z[i][nx+1:nx+nu,:])
    end

    return z, ψxs[2:end,:], ψus[2:end,:]
end

function validation(G::lti, Q::SystemlevelRENParams, w)
    Qe = REN(Q) # Construction from direct parameterization
    nx, nu = G.nx, G.nu # Dynamical system dimension
    batch = size(w[1], 2)
    ψxs = zeros(1, batch)
    ψus = zeros(1, batch)

    # Initialize the state of the REN
    x1 = zeros(nx, batch)
    h_1 = zeros(Qe.nx, batch)
    w_1 = zeros(nx, batch)
    hr0, ψr_1 = Qe(h_1, w_1)
    u_1 = zeros(nu, batch)

    # X0 = G(x1, u_1, w_1)
    wh0 = x1 - Qe.explicit.C2[1:nx,:]*hr0 .- Qe.explicit.by[1:nx,:]
    hr1, ψr0 = Qe(hr0, wh0)
    ψur0 = ψr0[nx+1:end,:]
    ψxr0 = ψr0[1:nx,:]
    
    # h1, ψ0 = ren(h0, wh0)
    
    X1 = (x1, hr1, ψur0, hr1)

    # function f(X_1, t)
    #     # Unpack the state of the REN from last time step
    #     x_1, h_1, w_1, u_1 = X_1 #system state, hidden state, ̂w, control input
    #     xt = G(x_1, u_1, w[t]) # system state at time t
    #     ht, ψ = Qe(h_1, w_1) # hidden state at time t
    #     # ̂wt = xt - ψt
    #     wht = xt - Qe.explicit.C2[1:nx,:]*ht .- Qe.explicit.by[1:nx,:]
    #     # wht = x_1 -ψ[1:nx,:]
    #     hnt, ψt= Qe(ht, wht) # ψ_x and \psi_u at time t
    #     # ψx = ψt[1:nx,:] # ψ_x at time t
    #     ut = ψt[nx+1:end, :] # ψ_u at time t
    #     Xt = (xt, ht, wht, ut) # Pack the state of the REN at time t
    #     zt = ψt
    #     return Xt, zt
    # end

    function f(X_1, t)
        # Unpack the state of the REN from last time step
        x_1, ht, u_1, hr = X_1 #system state, hidden state, ̂w, control input
        xt = G(x_1, u_1, w[t]) # system state at time t
        # ̂wt = xt - ψt
        hrt, ψrt = Qe(hr, w[t])
        wht = xt - Qe.explicit.C2[1:nx,:]*ht .- Qe.explicit.by[1:nx,:]
        # println(norm(wht-w[t]))
        hnt, ψt= Qe(ht, wht) 
        # println(norm(ψt-ψrt))
        x_t = ψt[1:nx,:] # ψ_x at time t
        # println(norm(ψrt[1:nx,:]-xt))
        # println(norm(xt-x_t))
        ut = ψt[nx+1:end, :] # ψ_u at time t
        # ut_ = tanh.(0.01*ut)
        Xt = (xt, hnt, ut, hrt) # Pack the state of the REN at time t
        # zt = ψt
        zt = vcat(xt ,ut)
        return Xt, zt
    end

    md = Flux.Recur(f,X1)
    z = md.(1:length(w))
 
    for i in 1:length(w)
        ψxs = vcat(ψxs, z[i][1:nx,:])
        ψus = vcat(ψus, z[i][nx+1:nx+nu,:])
    end

    return z, ψxs[2:end,:], ψus[2:end,:]
end