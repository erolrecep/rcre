package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type TreeNode struct {
	FeatureIdx uint32
	Threshold  float32
	LeftChild  uint32
	RightChild uint32
	LeafValue  float32
	IsLeaf     bool
}

type GBDTModel struct {
	Nodes       []TreeNode
	TreeRoots   []uint32
	NumTrees    int
	NumFeatures int
	BaseScore   float32
	IsBinary    bool
}

func LoadGBDTModel(path string) (*GBDTModel, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 24 || string(raw[:4]) != "RGBT" {
		return nil, fmt.Errorf("invalid RGBT header")
	}
	numTrees := int(binary.LittleEndian.Uint32(raw[4:8]))
	numNodes := int(binary.LittleEndian.Uint32(raw[8:12]))
	numFeatures := int(binary.LittleEndian.Uint32(raw[12:16]))
	baseScore := math.Float32frombits(binary.LittleEndian.Uint32(raw[16:20]))
	isBinary := binary.LittleEndian.Uint32(raw[20:24]) != 0

	expected := 24 + numTrees*4 + numNodes*24
	if len(raw) < expected {
		return nil, fmt.Errorf("truncated RGBT file: got %d need %d", len(raw), expected)
	}

	off := 24
	roots := make([]uint32, numTrees)
	for i := 0; i < numTrees; i++ {
		roots[i] = binary.LittleEndian.Uint32(raw[off : off+4])
		off += 4
	}

	nodes := make([]TreeNode, numNodes)
	for i := 0; i < numNodes; i++ {
		nodes[i] = TreeNode{
			FeatureIdx: binary.LittleEndian.Uint32(raw[off : off+4]),
			Threshold:  math.Float32frombits(binary.LittleEndian.Uint32(raw[off+4 : off+8])),
			LeftChild:  binary.LittleEndian.Uint32(raw[off+8 : off+12]),
			RightChild: binary.LittleEndian.Uint32(raw[off+12 : off+16]),
			LeafValue:  math.Float32frombits(binary.LittleEndian.Uint32(raw[off+16 : off+20])),
			IsLeaf:     binary.LittleEndian.Uint32(raw[off+20 : off+24]) != 0,
		}
		off += 24
	}

	return &GBDTModel{
		Nodes:       nodes,
		TreeRoots:   roots,
		NumTrees:    numTrees,
		NumFeatures: numFeatures,
		BaseScore:   baseScore,
		IsBinary:    isBinary,
	}, nil
}

func fastSigmoidf(z float32) float32 {
	if z > 16.0 {
		return 1.0
	}
	if z < -16.0 {
		return 0.0
	}
	return float32(1.0 / (1.0 + math.Exp(float64(-z))))
}

// PredictFlat runs inference over flat row-major [nRows * D] float32 buffer.
func (m *GBDTModel) PredictFlat(flat []float32, nRows int, out []float32) {
	D := m.NumFeatures
	nodes := m.Nodes
	roots := m.TreeRoots
	nTrees := m.NumTrees
	base := m.BaseScore
	isBin := m.IsBinary

	for i := 0; i < nRows; i++ {
		row := flat[i*D : (i+1)*D]
		score := base
		for t := 0; t < nTrees; t++ {
			curr := roots[t]
			for {
				node := &nodes[curr]
				if node.IsLeaf {
					score += node.LeafValue
					break
				}
				fidx := int(node.FeatureIdx)
				if fidx < D && row[fidx] <= node.Threshold {
					curr = node.LeftChild
				} else {
					curr = node.RightChild
				}
			}
		}
		if isBin {
			out[i] = fastSigmoidf(score)
		} else {
			out[i] = score
		}
	}
}
