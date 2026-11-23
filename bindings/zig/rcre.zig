// RCRE Zig Language Binding (C23 ABI)
const c = @cImport({
    @cInclude("rcre.h");
});

pub fn abiVersion() u32 {
    return c.rcre_abi_version();
}

pub fn halInit() void {
    c.rcre_hal_init();
}

pub fn cpuBrand() [*:0]const u8 {
    return c.rcre_cpu_get_brand();
}

pub fn arenaCreate(capacity: usize) ?*c.RCRE_Arena {
    return c.rcre_arena_create(capacity);
}

pub fn arenaDestroy(arena: *c.RCRE_Arena) void {
    c.rcre_arena_destroy(arena);
}

pub fn tensorCreate2D(rows: usize, cols: usize, arena: *c.RCRE_Arena) c.RCRE_Tensor {
    const shape = [_]usize{ rows, cols };
    return c.rcre_tensor_create(&shape[0], 2, c.RCRE_DTYPE_F32, c.RCRE_LAYOUT_TABULAR, arena);
}
