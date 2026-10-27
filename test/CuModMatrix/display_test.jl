function _check_cumod_display(A, type_name)
    compact = string(join(size(A), "×"), " ", type_name, " modulo ", A.N)

    array_text = repr(MIME"text/plain"(), Array(A))
    newline = findfirst('\n', array_text)
    contents =
        newline === nothing ? "" :
        String(SubString(array_text, nextind(array_text, newline)))
    text = isempty(contents) ? compact * ":" : compact * ":\n" * contents

    @test sprint(show, A) == compact
    @test repr(A) == text
    @test sprint(show, MIME"text/plain"(), A) == text
    @test "$(A)" == compact
    @test !occursin("0.0", text) # These fixtures are nonzero; padded storage must stay hidden.

    compact_io = IOBuffer()
    show(compact_io, A)
    @test String(take!(compact_io)) == compact

    text_io = IOBuffer()
    show(text_io, MIME"text/plain"(), A)
    @test String(take!(text_io)) == text

    captured = @capture_out begin
        show(compact_io, A)
        show(text_io, MIME"text/plain"(), A)
        sprint(show, A)
        repr(A)
        "$(A)"
    end
    @test isempty(captured)
end

function test_cumod_display()
    matrix = CuModMatrix(Float32[1 2 3; 4 5 6], 7)
    vector = CuModVector(Float32[1, 2, 3], 7)
    array = CuModArray(reshape(Float32[1, 2, 3, 4], 2, 1, 2), 7)

    _check_cumod_display(matrix, "CuModMatrix")
    _check_cumod_display(vector, "CuModVector")
    _check_cumod_display(array, "CuModArray")
end
