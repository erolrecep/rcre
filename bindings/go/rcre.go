package rcre

/*
#cgo CFLAGS: -std=c2x -I${SRCDIR}/../../include
#cgo darwin LDFLAGS: ${SRCDIR}/../../lib/librcre.a -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa -framework Metal -lm -lpthread
#cgo linux LDFLAGS: ${SRCDIR}/../../lib/librcre.a -lm -lpthread
#include "rcre.h"
static inline float rcre_go_dot_f32(const float* a, const float* b, size_t n) {
    return rcre_hal.dot_f32(a, b, n);
}
*/
import "C"
import "unsafe"

// ABIVersion returns packed RCRE C ABI version (e.g., 0x020000).
func ABIVersion() uint32 {
	return uint32(C.rcre_abi_version())
}

// InitHAL probes host CPU SIMD capabilities and initializes dispatch table.
func InitHAL() {
	C.rcre_hal_init()
}

// CPUBrand returns host processor brand string.
func CPUBrand() string {
	return C.GoString(C.rcre_cpu_get_brand())
}

// DotF32 executes SIMD-accelerated dot product using RCRE HAL.
func DotF32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return float32(C.rcre_go_dot_f32((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.size_t(n)))
}
