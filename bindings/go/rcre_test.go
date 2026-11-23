package rcre

import "testing"

func TestRCREGoBindings(t *testing.T) {
	InitHAL()
	if ABIVersion() < 0x020000 {
		t.Fatalf("unexpected ABI version: 0x%x", ABIVersion())
	}
	brand := CPUBrand()
	if brand == "" {
		t.Fatal("empty CPU brand")
	}
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	dot := DotF32(a, b)
	if dot != 70.0 {
		t.Fatalf("expected 70.0, got %f", dot)
	}
}
