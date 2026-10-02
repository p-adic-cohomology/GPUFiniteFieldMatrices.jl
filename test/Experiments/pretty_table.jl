using PrettyTables

function _pt_s(x)
    x === nothing && return ""
    x isa AbstractFloat && isnan(x) && return "NA"
    x isa AbstractFloat && return string(round(x; digits = 3))
    return string(x)
end

function print_and_save_table(
    rows,
    headers;
    csv_path::AbstractString,
    title::AbstractString = "",
)
    nrows = length(rows)
    ncols = length(headers)
    mat = Matrix{Any}(undef, nrows, ncols)

    for i = 1:nrows
        r = rows[i]
        for j = 1:ncols
            mat[i, j] = _pt_s(r[j])
        end
    end

    title != "" && println("\n--- $title (pretty) ---\n")
    title == "" && println("\n--- Results (pretty) ---\n")
    pretty_table(
        mat;
        column_labels = [headers],
        maximum_number_of_rows = -1,
        maximum_number_of_columns = -1,
        display_size = (-1, -1),
        fit_table_in_display_vertically = false,
        fit_table_in_display_horizontally = false,
    )

    title != "" && println("\n--- $title (Markdown) ---\n")
    title == "" && println("\n--- Results (Markdown) ---\n")
    pretty_table(
        mat;
        column_labels = [headers],
        backend = :markdown,
        maximum_number_of_rows = -1,
        maximum_number_of_columns = -1,
    )

    headers_csv = replace.(headers, "**" => "")
    open(csv_path, "w") do io
        println(io, join(headers_csv, ","))
        for i = 1:nrows
            fields = String[]
            for j = 1:ncols
                v = rows[i][j]
                s = _pt_s(v)
                if occursin(',', s) || occursin('"', s) || occursin('\n', s)
                    s = "\"" * replace(s, "\""=>"\"\"") * "\""
                end
                push!(fields, s)
            end
            println(io, join(fields, ","))
        end
    end

    println("\nSaved CSV: $csv_path")
    return nothing
end
