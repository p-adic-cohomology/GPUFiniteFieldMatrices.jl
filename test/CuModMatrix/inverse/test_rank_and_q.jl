function test_rank_and_q()
    N = 101
    Arank = [
        1 2 3 4
        1 2 3 4
        2 5 7 9
        3 6 8 10
    ]
    A1 = CuModMatrix(Arank, N)
    F1 = pluq(A1)
    @test F1.rank < 4
    Acol = [
        0 2 3
        0 5 7
        0 1 4
    ]
    A2 = CuModMatrix(Acol, N)
    F2 = pluq(A2)
    @test F2.q != [1, 2, 3]
end
