using Parameters
using NearestNeighbors
using LinearAlgebra
using DataStructures
using LibGEOS
using ProgressMeter
using PyCall

abstract type SpaceColonizationPoint end

@with_kw struct Node <: SpaceColonizationPoint
    x::Float64
    y::Float64
    parent::Union{Nothing,Node} = nothing
end

@with_kw mutable struct Attractor <: SpaceColonizationPoint
    x::Float64
    y::Float64
    active::Bool = true
end

# type this?
# KNN requires the points as columns
function p2a(points)
    location = Array{Float64,2}(undef, 2, length(points))
    for (i, point) in enumerate(points)
        @inbounds location[:,i] = [point.x, point.y]
    end
    return location
end

function node_from_attractors(node::Node, attractors::Vector{Attractor}, length::Number=1)
    attractor_vector = sum(p2a(attractors) .- [node.x, node.y], dims=2)
    attractor_vector /= norm(attractor_vector)
    attractor_vector *= length
    newx, newy = [node.x, node.y] + attractor_vector
    return Node(newx, newy, node)
end

function find_influenced_nodes(nodes::Vector{Node}, attractors::Vector{Attractor}, attraction_distance::Number)
    # node_array = p2a(nodes)
    # dftree = DataFreeTree(KDTree, node_array; leafsize=50)
    # nodes_kdtree = injectdata(dftree, node_array)
    nodes_kdtree = KDTree(p2a(nodes); leafsize=40, reorder=false)
    nearest_nodes, node_distances = nn(nodes_kdtree, p2a(attractors))
    influenced_nodes = DefaultDict{Node,Vector{Attractor}}([]) 
    for (attractor_ix, (node_ix, dist)) in enumerate(zip(nearest_nodes, node_distances))
        dist <= attraction_distance && push!(influenced_nodes[nodes[node_ix]], attractors[attractor_ix])
    end
    return influenced_nodes
end

function find_deactivated_attractor_ids(nodes::Vector{Node}, attractor_kdtree::KDTree, kill_distance::Number)
    attractor_kill_candidates = inrange(attractor_kdtree, p2a(nodes), kill_distance)
    attractor_ids = unique(reduce(vcat, attractor_kill_candidates))
    return attractor_ids
end

function kill_attractors_by_id!(attractor_ids::Vector{Int}, attractors::Vector{Attractor})
    killed_attractors = (attractors[i] for i in attractor_ids)
    setproperty!.(killed_attractors, :active, false)
end

function get_sketch_dims(size::String)
    vsketch = pyimport("vsketch")
    vsk = vsketch.Vsketch()
    vsk.size(size)
    return vsk.width, vsk.height
end

W, H = get_sketch_dims("12x9in")
const attraction_distance = 100
const kill_distance = 5
const segment_length = 1


attractors = [Attractor(x, y, true) for (x, y) in eachrow(rand(5000, 2) .* [W H])]
nodes = [Node(5, 5, nothing), Node(W - 5, H - 5, nothing)]

attractor_kdtree = KDTree(p2a(attractors); leafsize=40, reorder=false)

active_attractors = [a for a in attractors if a.active]

lines = Vector{Vector{Float64}}[]
ITERATIONS = 1600
@showprogress 1 "Running..." for i in 1:ITERATIONS
    influenced_nodes = find_influenced_nodes(nodes, active_attractors, attraction_distance)

    new_nodes = Node[]
    for (node, influencing_attractors) in pairs(influenced_nodes)
        new_node = node_from_attractors(node, influencing_attractors, segment_length)
        push!(new_nodes, new_node)
        push!(lines, [[node.x, node.y], [new_node.x, new_node.y]])
    end
        
    if length(new_nodes) == 0
        @info("No New Nodes Created iter=$i")
        break
    end

    push!(nodes, new_nodes...)

    deactivated_ids = find_deactivated_attractor_ids(new_nodes, attractor_kdtree, kill_distance)
    kill_attractors_by_id!(deactivated_ids, attractors)

    active_attractors = [a for a in attractors if a.active]
    if length(active_attractors) == 0
        @info("No active Attractors iter=$i")
        break
    end
end
@info "Done"



# lines = runmodel(100.0, 3.0, 1)
mls = MultiLineString(lines)
border = LinearRing([[0,0], [W, 0], [W,H], [0, H], [0, 0]])
elements = []
push!(elements, mls)
push!(elements, border)

Plots.plot([border, mls], yflip=true)
