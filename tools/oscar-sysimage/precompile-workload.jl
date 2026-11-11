using GPUFiniteFieldMatrices

# This exercises only CPU-backed representation/conversion helpers. Do not
# allocate CuArrays, query a device, or create a CUDA context in this workload.
low = Int[1 2; 3 4]
high = Int[0 1; 1 0]
value = KaratsubaMatrix(
    low,
    high;
    low_modulus = 11,
    high_modulus = 11,
    modulus = 11,
)
expected = low .+ 11 .* high
from_karatsuba(Int, value) == expected || error("host Karatsuba workload returned an unexpected result")
