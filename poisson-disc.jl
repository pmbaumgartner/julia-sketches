using Distances
using StaticArrays
using LinearAlgebra

struct PoissonDiscSample
    area::Vector{Float64}
    radius::Float64
    k::Int
    cell_size::Float64
    n_dims::Int
    grid_dims::Vector{Int}
    grid::Array
    neighbor_movement::Vector{Vector{Int64}}
end

function PoissonDiscSample(area, radius, k)
    n_dims = length(area)
    cell_size = radius / sqrt(n_dims)
    grid_dims = ceil.(Int, area / cell_size)
    GridVector = SVector{n_dims,Float64}
    grid = fill(GridVector(repeat([NaN], n_dims)), grid_dims...)::Array{GridVector,n_dims}
    neighbor_movement =
        collect.(vec(collect(Iterators.product([[-2, -1, 0, 1, 2] for _ in 1:n_dims]...))))
    return PoissonDiscSample(
        area, radius, k, cell_size, n_dims, grid_dims, grid, neighbor_movement
    )
end

isassigned(vector) = !all(isnan(e) for e in vector)
function getpoints(pdsample::PoissonDiscSample)
    return convert.(Ref(Vector), filter(isassigned, unique(pdsample.grid)))
end
grid_index(vector, pdsample::PoissonDiscSample) = Int.(vector .÷ pdsample.cell_size) .+ 1
function view_index(grid_ix, pdsample::PoissonDiscSample)
    @inbounds @views pdsample.grid[grid_ix...]
end
function index_assigned(grid_ix, pdsample::PoissonDiscSample)
    return isassigned(view_index(grid_ix, pdsample))
end

function inbounds(vector, pdsample::PoissonDiscSample)
    for (i, e) in enumerate(vector)
        @inbounds (e < 0 || e > pdsample.area[i]) && return false
    end
    return true
end

function inbounds_grid(grid_location, pdsample::PoissonDiscSample)
    for (i, d) in enumerate(grid_location)
        @inbounds (d < 1 || d > pdsample.grid_dims[i]) && return false
    end
    return true
end

function insert_sample!(vector, pdsample::PoissonDiscSample)
    @inbounds pdsample.grid[grid_index(vector, pdsample)...] = vector
end

function newvector(source_vector, pdsample::PoissonDiscSample; minrrange=1, R=2)
    nv = rand(length(source_vector)) .- 0.5
    nv /= norm(nv)
    r = map(x -> minrrange + x * (R - minrrange), rand()) * pdsample.radius
    nv *= r
    return source_vector + nv
end

function valid_grid_generator(vector, pdsample::PoissonDiscSample)
    vector_ix = grid_index(vector, pdsample)
    grid_gen = (vector_ix + n for n in pdsample.neighbor_movement)
    grid_locations = Iterators.filter(x -> inbounds_grid(x, pdsample), grid_gen)
    return grid_locations
end

function far(vector, pdsample::PoissonDiscSample)
    grid_locations = valid_grid_generator(vector, pdsample)
    for index in grid_locations
        v = view_index(index, pdsample)
        !isassigned(v) && continue
        euclidean(v, vector) < pdsample.radius && return false
    end
    return true
end

function find_candidate(vector, pdsample::PoissonDiscSample; R=2)
    for _ in 1:(pdsample.k)
        candidate_vector = newvector(vector, pdsample; R=R)
        inbounds(candidate_vector, pdsample) &&
            far(candidate_vector, pdsample) &&
            return (true, candidate_vector)
    end
    return (false, vector)
end

function bridson_sample(area, radius, k; R=2)
    pdsample = PoissonDiscSample(area, radius, k)
    active_list = Vector{Float64}[]
    initial_sample = pdsample.area .* rand(pdsample.n_dims)
    push!(active_list, initial_sample)
    insert_sample!(initial_sample, pdsample)
    while !isempty(active_list)
        sample = rand(active_list)
        success, candidate = find_candidate(sample, pdsample; R=R)
        if success
            push!(active_list, candidate)
            insert_sample!(candidate, pdsample)
        else
            filter!(x -> x != sample, active_list)
        end
    end
    return getpoints(pdsample)
end