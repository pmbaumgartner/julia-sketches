using NearestNeighbors
using LinearAlgebra
using DataStructures
using ProgressMeter

abstract type SpaceColonizationPoint end

struct Node <: SpaceColonizationPoint
    x::Float64
    y::Float64
    parent::Union{Nothing,Node}
end
Node(x, y) = Node(x, y, nothing)

mutable struct Attractor <: SpaceColonizationPoint
    x::Float64
    y::Float64
    active::Bool
    nearest_node::Union{Nothing,Node}
    nn_dist::Float64
end
Attractor(x, y) = Attractor(x, y, true, nothing, Inf)

# KNN requires the points as columns
function p2a(points)
    location = Array{Float64,2}(undef, 2, length(points))
    @inbounds location[1, :] = [p.x for p in points]
    @inbounds location[2, :] = [p.y for p in points]
    return location
end

function node_from_attractors(node::Node, attractors::Vector{Attractor}, length::Number=1)
    attractor_vector = sum(p2a(attractors) .- [node.x, node.y]; dims=2)
    attractor_vector /= norm(attractor_vector)
    attractor_vector *= length
    newx, newy = [node.x, node.y] + attractor_vector
    return Node(newx, newy, node)
end

function update_attractor_nn!(
    nodes::Vector{Node}, attractors::Vector{Attractor}, attraction_distance::Float64
)
    nodes_kdtree = KDTree(p2a(nodes); leafsize=40, reorder=false)
    nearest_nodes, node_distances = nn(nodes_kdtree, p2a(attractors))
    attractor_ix_with_influence = findall(x -> x <= attraction_distance, node_distances)
    for attractor_ix in attractor_ix_with_influence
        attractor = attractors[attractor_ix]
        closer_node = node_distances[attractor_ix] < attractor.nn_dist
        if closer_node
            attractor.nearest_node = nodes[nearest_nodes[attractor_ix]]
            attractor.nn_dist = node_distances[attractor_ix]
            # attractors[attractor_ix] = attractor
        end
    end
    return attractors
end

function find_influenced_nodes(attractors::Vector{Attractor})
    influenced_nodes = DefaultDict{Node,Vector{Attractor}}([])
    for attractor in attractors
        nearest_node = attractor.nearest_node
        push!(influenced_nodes[nearest_node], attractor)
    end
    return influenced_nodes
end

function find_deactivated_attractor_ids(
    nodes::Vector{Node}, attractor_kdtree::KDTree, kill_distance::Float64
)
    attractor_kill_candidates::Vector{Vector{Int64}} = inrange(
        attractor_kdtree, p2a(nodes), kill_distance
    )
    attractor_ids = unique(reduce(vcat, attractor_kill_candidates))
    return attractor_ids
end

function kill_attractors_by_id!(attractor_ids::Vector{Int}, attractors::Vector{Attractor})
    killed_attractors = (attractors[i] for i in attractor_ids)
    return setproperty!.(killed_attractors, :active, false)
end

function space_colonization(
    attractor_positions::Vector{Vector{Float64}},
    node_positions::Vector{Vector{Float64}},
    attraction_distance::Float64,
    kill_distance::Float64,
    segment_length::Float64;
    max_iterations=10000,
)
    attractors = [Attractor(coords...) for coords in attractor_positions]
    nodes = [Node(coords...) for coords in node_positions]
    new_nodes = copy(nodes)
    active_attractors = copy(attractors)

    attractor_kdtree = KDTree(p2a(attractors); leafsize=40, reorder=false)

    lines = Vector{Vector{Float64}}[]
    @showprogress 1 "Running..." for i in 1:max_iterations
        update_attractor_nn!(new_nodes, active_attractors, attraction_distance)
        attractors_with_nn = [
            attractor for
            attractor in active_attractors if !isnothing(attractor.nearest_node)
        ]

        if length(attractors_with_nn) == 0
            @info("No Attractors with Nodes in Range (iter=$i)")
            break
        end

        influenced_nodes = find_influenced_nodes(attractors_with_nn)

        new_nodes = Node[]
        for (node, influencing_attractors) in pairs(influenced_nodes)
            new_node = node_from_attractors(node, influencing_attractors, segment_length)
            push!(new_nodes, new_node)
            push!(lines, [[node.x, node.y], [new_node.x, new_node.y]])
        end

        push!(nodes, new_nodes...)

        deactivated_ids = find_deactivated_attractor_ids(
            new_nodes, attractor_kdtree, kill_distance
        )
        kill_attractors_by_id!(deactivated_ids, attractors)

        active_attractors = [a for a in attractors if a.active]
        if length(active_attractors) == 0
            @info("No active Attractors (iter=$i)")
            break
        end
    end
    return lines
end
