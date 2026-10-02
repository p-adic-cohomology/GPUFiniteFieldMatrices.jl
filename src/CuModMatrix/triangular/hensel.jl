"""
    hensel_pseudoinverse(N, precision, A, T)

Hensel lifts mod N solution T to the linear system AX-I=0 to mod N^precision
Uses the CuModMatrix type.

INPUTS:
* "N" -- integer, a prime number 
* "precision" -- integer 
* "A" -- matrix, integer coefficients
* "T" -- matrix, integer coefficients, satisfies AT-I=0 mod N
"""
function hensel_pseudoinverse(N, precision, A, T)
    i = 1
    while i < precision
        T = 2*T - T * (A*T)
        i *= 2
    end
    R, pi = residue_ring(ZZ, N^precision)
    return matrix(R, [R(x) for x in Array(T)])
end
