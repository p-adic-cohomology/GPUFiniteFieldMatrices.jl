@testset "Backend test driver selection" begin
    parse_backend = BackendTestDriver.parse_backend_selector
    optional_backends = BackendTestDriver.optional_backends

    @test parse_backend(String[]) === nothing
    @test parse_backend(["--backend=metal"]) === :metal
    @test parse_backend(["--backend=cuda"]) === :cuda
    @test_throws ArgumentError parse_backend(["--backend=cpu"])
    @test_throws ArgumentError parse_backend(["--backend=cuda", "--backend=metal"])

    package_paths = Dict("CUDA" => "/tmp/CUDA.jl", "Metal" => "/tmp/Metal.jl")
    package_available = name -> get(package_paths, name, nothing)
    @test optional_backends(nothing, package_available) == (:cuda, :metal)
    @test optional_backends(:metal, package_available) == (:metal,)
    @test optional_backends(:cuda, package_available) == (:cuda,)
    @test_throws ArgumentError optional_backends(:cuda, name -> nothing)
end
