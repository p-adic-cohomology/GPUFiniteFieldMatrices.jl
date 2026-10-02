using LinearAlgebra

"""
    count_flops(A::Matrix,B::Matrix)

Counts the total number of floating point operations, including mod.
According to GFlops, Julia's mod operation takes only 1 flop.
This is adjustable through the variable MOD_OPS.
"""
function count_flops(A, B)
    MOD_OPS = 1

    m, n = size(A)
    n, k = size(B)

    # Each element takes n multiplications and n-1 additions
    # There are mk total number of elements
    return m * k * (2n - 1)
end

"""
    count_Flops(A::Matrix,B::Matrix,TILE_WIDTH,MAX_OPS)

Counts the total number of floating point operations, excluding mod.
"""
function count_Flops(A, B, TILE_WIDTH, MAX_OPS)
    MOD_OPS = 1

    m, n = size(A)
    n, k = size(B)

    # Each element takes n multiplications and n-1 additions
    # There are mk total number of elements

    # If MAX_OPS is larger than the number of multiplications
    if MAX_OPS >= n
        # Then the algorithm only mods once, at the end
        return m * k * (2n - 1 + MOD_OPS)
        # If MAX_OPS is larger than each tile
    elseif MAX_OPS >= TILE_WIDTH
        # Then we mod at the end of the tile
        return m * k * (2n - 1 + (floor(n/TILE_WIDTH)+1) * MOD_OPS)
        # Otherwise MAX_OPS can happen multiple times a tile
    else
        # So each element, in each tile (there are floor(n,TILE_WIDTH)+1 tiles)
        # is modded floor(TILE_WDITH,MAX_OPS) number of times.
        return m *
               k *
               (2n - 1 + (floor(n/TILE_WIDTH)+1) * MOD_OPS * (floor(TILE_WIDTH, MAX_OPS)+1))
    end
end
