using CUDA, LinearAlgebra

function mat_mul_plain(d_A, d_B, P)
    """
    Using default implementation of MatMul.
    """

    d_C = d_A * d_B
    return d_C
end
