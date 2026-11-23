module RCRE

export abi_version, hal_init, cpu_brand, arena_create, arena_destroy

const LIBRCRE = normpath(joinpath(@__DIR__, "..", "..", "lib",
    Sys.isapple() ? "librcre.dylib" : "librcre.so"))

abi_version()::UInt32 = ccall((:rcre_abi_version, LIBRCRE), UInt32, ())
hal_init() = ccall((:rcre_hal_init, LIBRCRE), Cvoid, ())
cpu_brand()::String = unsafe_string(ccall((:rcre_cpu_get_brand, LIBRCRE), Cstring, ()))
arena_create(cap::Csize_t)::Ptr{Cvoid} = ccall((:rcre_arena_create, LIBRCRE), Ptr{Cvoid}, (Csize_t,), cap)
arena_destroy(arena::Ptr{Cvoid}) = ccall((:rcre_arena_destroy, LIBRCRE), Cvoid, (Ptr{Cvoid},), arena)

end
