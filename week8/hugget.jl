# this time we use a different method of writing code

using Plots 
using Parameters, Roots
using LinearAlgebra, Statistics, Distributions, Printf, Random





# this function creates a grid
function asset_grid(;N=120,a_min=0.0,a_max=2.0)
    a_grid = collect(range(a_min, stop = a_max, length = N))
    return a_grid
end

# this is the key function here, it solves the consumption-savings problem

function vfi_infinite(hh, a_grid; ϵ = 1e-6, maxiter = 300)
    y_grid = hh.y
    P      = hh.P
    
    N_a    = length(a_grid);
    N_y    = length(y_grid);
    V      = Matrix{Float64}(undef, N_a, N_y);
    V_old  = Matrix{Float64}(undef, N_a, N_y);
    a′     = Matrix{Float64}(undef, N_a, N_y);
    a′_ind = Matrix{Int}(undef, N_a, N_y);
    c      = Matrix{Float64}(undef, N_a, N_y);
    w_temp = Matrix{Float64}(undef, N_a, 1);
    u_temp = Matrix{Float64}(undef, N_a, 1);
    c_temp = Matrix{Float64}(undef, N_a, 1);
    
    
    
    # guess something initial
    V_old    .= 0.0
    err      = 10.0
    
    # start iteration
    for n in 1:maxiter
        
    
        # for each possible realization of income today...
        for iy in eachindex(y_grid)
            # for each asset grid point today...
    
            for ia in eachindex(a_grid)
    
                #... consider each possible asset grid point tomorrow
                for ia′ in eachindex(a_grid)
    
                    #... calculate what would be consumption today
    
                    if (1+hh.r) * a_grid[ia] + y_grid[iy] - a_grid[ia′] < 1e-10
                        c_temp[ia′] = 1e-10  # if consumption is negative, set to a very small number
                        u_temp[ia′] = -Inf  # calculate utility
                    else
                        c_temp[ia′] = (1+hh.r) * a_grid[ia] + y_grid[iy] - a_grid[ia′]
                        u_temp[ia′] = hh.u(c_temp[ia′])  # calculate utility
                    end
    
                    #... utility  calculate expected continuation value 
    
                    w_temp[ia′] = u_temp[ia′] + hh.β * sum(P[iy,:] .* V_old[ia′,:])  # calulate utility + EXPECTED continuation value
                    
                end
    
                V[ia,iy], ia′_opt  = findmax(w_temp[:])  # find optimum - stores Value und policy (index of optimal choice)
                a′[ia,iy]            = a_grid[ia′_opt]    # record optimal assets tomorrow
                a′_ind[ia,iy]        = ia′_opt
                
            end
        end
    
    
        err,z = findmax(abs.(V .- V_old));
    
        if err < ϵ
            break
        end
    
        V_old .= V 
    
    end
    c = (1+hh.r) * a_grid .+ y_grid - a′
    return (V, a′, a′_ind, c, err)
end


# this creates a discretized grid for income
struct rouwenhurst
    # Define output
    y::Vector{Float64} # Grid for income
    P::Matrix{Float64} # Transition probability matrix
    π_vec::Vector{Float64} # Stationary probability mass function
    # Define function
    # this is a so called "inner constructor method"
    function rouwenhurst(n, μ, ρ, σ)
        # Transitition matrix parameter
        p = (1 + ρ) / 2
        # Width parameter
        ψ = sqrt((n - 1) * σ^2 / (1 - ρ^2))
        # Grid for income (before normalization)
        y = collect(range((μ - ψ), stop = (μ + ψ), length = n))
        # Transition probability matrix for π=2
        P_temp = [p (1-p); (1-p) p]
        # Two cases for n (do not do anything if n = 2, do things if n > 2)
        if n == 2
            P = P_temp
        elseif n > 2
            for i = 1:(n-2)
                # (n-1) vector of zeros
                zeros_vec = zeros(size(P_temp, 2))
                # Update transititon probability matrix
                P_temp =
                    (p * [P_temp zeros_vec; zeros_vec' 0]) +
                    ((1 - p) * [zeros_vec P_temp; 0 zeros_vec']) +
                    ((1 - p) * [zeros_vec' 0; P_temp zeros_vec]) +
                    (p * [0 zeros_vec'; zeros_vec P_temp])
            end
            # Ensure elements in each row sum to one
            P = Matrix{Float64}(undef, n, n)
            P[1, :] = P_temp[1, :]
            P[n, :] = P_temp[n, :]
            for r = 2:(n-1)
                P[r, :] = P_temp[r, :] ./ sum(P_temp[r, :])
            end
        end
        # Stationary probability mass function
        π_vec = (ones(n) ./ n)'
        for i = 1:1000
            π_vec= π_vec * (P^i)
        end
        # Convert into a column vector
        π_vec = π_vec'
        # Return output - this "new" is special to inner constructor
        new(y, P, π_vec)
    end
end;


# this function obtains the stationary distribution
function stationary_dist(hh, a_grid, a′_ind)
    y_grid = hh.y
    P      = hh.P
    
    N_a    = length(a_grid);
    N_y    = length(y_grid);
    
    N_ay   = N_a * N_y

    # start with a uniform
    g_vec  = ones(N_ay)' ./ N_ay

    # build a big transition matrix
    P_big = zeros(N_ay,N_ay)
    for j in 1:1:N_y
        for j′ in 1:1:N_y
            for i in 1:1:N_a
            P_big[i+ ((j-1)*N_a),a′_ind[i,j]+((j′-1)*N_a)] = 1.0 * P[j,j′]
            end
        end
    end

    g_vec = g_vec * P_big ^ 10000
    g_mat = reshape(g_vec,N_a,N_y)
end



function get_asset_demand(a_grid,rate,P_rouwenhurst,y_rouwenhurst)
    hh   = Household(r=rate,P=P_rouwenhurst,y=y_rouwenhurst)
    V, a′, a′_ind, c, err = vfi_infinite(hh, a_grid)
    g_mat = stationary_dist(hh, a_grid, a′_ind)
    asset_demand = sum(a_grid .* g_mat)
end




Household = @with_kw (r = 0.01,                 
σ = 2.0,
β = 0.98,
y = [0.8 1.2],
P = [0.7 0.3; 0.3 0.7],
u = σ == 1 ? x -> log(x) : x -> (x^(1 - σ) - 1) / (1 - σ)
)

n = 2;
μ = 1;
ρ = 0.9;
σ = 0.05;
rouwenhurst_AR = rouwenhurst(n, μ, ρ, σ);
P_rouwenhurst = rouwenhurst_AR.P;
y_rouwenhurst = rouwenhurst_AR.y;
π_vec_rouwenhurst = rouwenhurst_AR.π_vec;

a_grid = asset_grid(a_min = -0.3,a_max = 4, N = 100)
asset_demand(r) = get_asset_demand(a_grid,r,P_rouwenhurst,y_rouwenhurst')

y_for_plot = LinRange(0.00,0.02,50)
x_for_plot = asset_demand.(y_for_plot)

plot(x_for_plot,y_for_plot,xlabel = "Savings", ylabel = "Interest rate", legend = false)


r_eqm = find_zero(asset_demand, (-0.1,  0.02))


