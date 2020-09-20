#script for creating Octree grids for POLARIS
#right now a clumpy AGN-Torus is modelled. The underlying continous model is a wedge rotated around the z-axis
#with density(R) = rho_0 * R**-0.5

using PyCall		#using python structs for easy IO compatible with POLARIS
s = pyimport("struct")
using Distributed	#the children of a cell don't depend on each other -> parallelize
addprocs(4)    		#right amount of processes on my machine, should probably be abstracted 
using ArgParse
using Random

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "-N"
            help = "Number of Clumps in Torus"
            arg_type = Int
            default = 10^3
        "-S"
            help = "Seed for random generator"
            arg_type = Int
            default = 0
        "-R"
            help = "Clump radius in pc"
            arg_type = Float64
            default = 1.0
        end
    return parse_args(s)
end

@everywhere mutable struct cell
    x::Float64
    y::Float64
    z::Float64

    length::Float64
    level::UInt8
    isleaf::UInt8
    data::Array{Float64}
    branches::Array{cell}
end

@everywhere mutable struct octree
    root::cell
    mass::Float64
    maxlevel::UInt8
	minlevel::UInt8
    cellcounter
    totalmass
end

@everywhere function getcenter(c::cell)::Array{Float64}
    cx = c.x + 0.5 * c.length
    cy = c.y + 0.5 * c.length
    cz = c.z + 0.5 * c.length
    return [cx,cy,cz]
end

@everywhere function density(pos::Vector)::Float64
    dens = 0.0
	#problem with the model: clumps cover the the whole center for certain configurations
	#this ensures that a cylinder in z-direction is left free
	if sqrt(pos[1]^2 + pos[2]^2) < Rᵢ 
        return 0.0
    end
    for i =1:length(view(positions,:,1))
        if distance(pos,view(positions,i,:)) <= R_cl
            dens += 1 #whole model gets normalized while writing the tree, so relativ density is fine here
        end
    end
    return dens
end

#current way of deciding if a cell is filled or split up:
#calculate mass in cell, calculate mass that would result if cell is split up again
#if difference is small enough fill cell
#cells near the center always get split up to the maximum level
@everywhere function refinement(c::cell)
	pos = getcenter(c)
	if sqrt(pos[1]^2 + pos[2]^2 + pos[3]^2) <= 3.0 * con_pc
		return false
	else
		mass = density(pos) * (sl/(2^(c.level)))^3
		massNextLevel = 0.
		l = c.length * 0.5
		level = c.level + 1
		v = (sl/(2^(level)))^3
		x = c.x
		y = c.y
		z = c.z
		massNextLevel += density([x + 0.5*l,y + 0.5*l,z + 0.5*l]) * v
		massNextLevel += density([x + l + 0.5*l,y + 0.5*l,z + 0.5*l]) * v
		massNextLevel += density([x + 0.5*l,y + l + 0.5*l,z + 0.5*l]) * v
		massNextLevel += density([x + l + 0.5*l,y + l + 0.5*l,z + 0.5*l]) * v
		massNextLevel += density([x + 0.5*l,y + 0.5*l,z + l + 0.5*l]) * v
		massNextLevel += density([x + l + 0.5*l,y + 0.5*l,z + l + 0.5*l]) * v
		massNextLevel += density([x + 0.5*l,y + l + 0.5*l,z + l + 0.5*l]) * v
		massNextLevel += density([x + l + 0.5*l,y + l + 0.5*l,z + l + 0.5*l]) * v
		if massNextLevel == 0.
			return true
		else
			return abs(massNextLevel - mass) / massNextLevel < 10^-4
		end
	end
end

@everywhere function distance(pos1::Vector,pos2)::Float64
    return sqrt((pos1[1] - pos2[1])^2 + (pos1[2] - pos2[2])^2 + (pos1[3] - pos2[3])^2)
end

@everywhere function fillcell(c::cell)::Float64
	mass = density(getcenter(c)) * c.length^3
    push!(c.data,mass)
    return mass
end

@everywhere function addNextLevel(c::cell,t::octree)
	if c.level == t.maxlevel || (c.level >= t.minlevel && refinement(c))
        c.isleaf = 1
        Threads.atomic_add!(t.cellcounter, 1)
        Threads.atomic_add!(t.totalmass, fillcell(c))
        if t.cellcounter[] % 1000 == 0
            println(t.cellcounter)
        end

    else
        x = c.x
        y = c.y
        z = c.z
        level = c.level + 1
        l = c.length * 0.5
        c1 = cell(x,y,z,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c1)
        c2 = cell(x + l,y,z,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c2)
        c3 = cell(x,y + l,z,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c3)
        c4 = cell(x + l,y + l,z,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c4)
        c5 = cell(x,y,z + l,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c5)
        c6 = cell(x + l,y,z + l,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c6)
        c7 = cell(x,y + l,z + l,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c7)
        c8 = cell(x + l,y + l,z + l,l,level,0,Vector{Float64}(),Vector())
        push!(c.branches, c8)

        Threads.@threads for i=1:8
            addNextLevel(c.branches[i],t)
        end
    end
end

function writeTree(t::octree,c::cell,io)
    write(io,s.pack("H",c.isleaf))
    write(io,s.pack("H",c.level))
    if c.isleaf == 1
        gas_m = c.data[1]*t.mass /t.totalmass[]
        gas_d = gas_m / c.length^3
        write(io,s.pack("f",gas_d))
        for i in 1:15
            write(io,s.pack("f",gas_d * 0.01 * dust_fractions[i]))
        end
    else
        for i=1:8
            writeTree(t,c.branches[i],io)
        end
    end
end

###############---MAIN---###############

#Constants
const con_pc = 3.0856775814671916e+16
const con_msun = 1.9884754153381438e+30

#grid constants
const grid_id = 20
const nr_dust_densities = 15
const data_id = ones(Int,nr_dust_densities) .* 29
const data_len = length(data_id) + 1
const dust_fractions = [0.0492868,0.01971472,0.00985736,0.07288324,0.02915329,0.01457665,0.10777664,0.04311066,0.02155533,0.15937553,0.06375021,0.03187511,0.23567779,0.09427111,0.04713556]

#commandline arguments
parsed_args = parse_commandline()
const N = parsed_args["N"]
const S = parsed_args["S"]
const R_cl = parsed_args["R"] * con_pc
const outR = round(Int,parsed_args["R"])
Random.seed!(S)

#define modell parameters here
const sl = 100 * con_pc
const maxlevel = 8
const minlevel = 6
const modellmass = 2236036 * con_msun

#choose positions
const Rₒ = sl * 0.5
const Rᵢ = 0.4 * con_pc
const positions = zeros(Float64,N,3)

for i in 1:N
    θ = acos(2*rand() - 1)
    while true
        if θ >= π/4 && θ <= 3π/4
            break
        else
            θ = acos(2*rand() - 1)
        end
    end
    ϕ = 2π * rand()
    r = (rand() * ( Rₒ^(5/2) - Rᵢ^(5/2)) + Rᵢ^(5/2))^(2/5)

    positions[i,1] = r*sin(θ)cos(ϕ)
    positions[i,2] = r*sin(θ)sin(ϕ)
    positions[i,3] = r*cos(θ)
end

#create initial values
tmpmass = 0
cellcounter = Threads.Atomic{Int}(0)
totalmass = Threads.Atomic{Float64}(0.0)
root = cell(-0.5*sl,-0.5*sl,-0.5*sl,sl,0,0,Vector{Float64}(),Vector())
tree = octree(root,modellmass,maxlevel,minlevel,cellcounter,totalmass)

#build and write tree
addNextLevel(root,tree)
open("bigclumps_$(N).dat", "w") do io
    write(io,s.pack("H",grid_id))
    write(io,s.pack("H",data_len))
    write(io,s.pack("H",28))
    for i=1:data_len - 1
        write(io,s.pack("H",data_id[i]))
    end
    write(io,s.pack("d",sl))
    writeTree(tree,tree.root,io)
end
