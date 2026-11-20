module GFMPerfSetupPaths

export same_source_path

"""Compare existing source directories by physical path, including symlink aliases."""
function same_source_path(recorded, expected)
    recorded isa AbstractString && expected isa AbstractString || return false
    isdir(recorded) && isdir(expected) || return false
    return realpath(recorded) == realpath(expected)
end

end
