/**
 * RCRE WebGPU Compute Bridge & WASM Memory Interop (Phase 5)
 * Dispatches SGEMM workloads onto WebGPU WGSL shaders from rcre.wasm views.
 */

export class RCREWebGPU {
    constructor(device, pipeline) {
        this.device = device;
        this.pipeline = pipeline;
    }

    static async init() {
        if (!navigator.gpu) return null;
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return null;
        const device = await adapter.requestDevice();

        const shaderCode = `
            struct Params { M: u32, N: u32, K: u32, _pad: u32 }
            @group(0) @binding(0) var<storage, read> A: array<f32>;
            @group(0) @binding(1) var<storage, read> B: array<f32>;
            @group(0) @binding(2) var<storage, read_write> C: array<f32>;
            @group(0) @binding(3) var<uniform> p: Params;

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let row = gid.y;
                let col = gid.x;
                if (row >= p.M || col >= p.N) { return; }
                var acc: f32 = 0.0;
                for (var k: u32 = 0u; k < p.K; k = k + 1u) {
                    acc = acc + A[row * p.K + k] * B[k * p.N + col];
                }
                C[row * p.N + col] = acc;
            }
        `;

        const module = device.createShaderModule({ code: shaderCode });
        const pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });
        return new RCREWebGPU(device, pipeline);
    }

    async gemmF32(A, B, M, N, K) {
        const makeBuf = (arr, usage) => {
            const buf = this.device.createBuffer({
                size: arr.byteLength,
                usage,
                mappedAtCreation: true
            });
            new Float32Array(buf.getMappedRange()).set(arr);
            buf.unmap();
            return buf;
        };

        const bufA = makeBuf(A, GPUBufferUsage.STORAGE);
        const bufB = makeBuf(B, GPUBufferUsage.STORAGE);
        const bufC = this.device.createBuffer({
            size: M * N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const bufP = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM,
            mappedAtCreation: true
        });
        new Uint32Array(bufP.getMappedRange()).set([M, N, K, 0]);
        bufP.unmap();

        const readback = this.device.createBuffer({
            size: M * N * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const bg = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufA } },
                { binding: 1, resource: { buffer: bufB } },
                { binding: 2, resource: { buffer: bufC } },
                { binding: 3, resource: { buffer: bufP } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
        pass.end();
        enc.copyBufferToBuffer(bufC, 0, readback, 0, M * N * 4);
        this.device.queue.submit([enc.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        bufA.destroy(); bufB.destroy(); bufC.destroy(); bufP.destroy(); readback.destroy();
        return out;
    }
}
