package engine

import (
	"bytes"
	"math"
	"os"
	"sort"
)

type SearchHit struct {
	DocID uint32
	Score float32
}

type BM25Index struct {
	NumDocs       uint32
	AvgDocLen     float32
	K1            float32
	B             float32
	DocLens       []uint32
	Vocab         map[string]uint32
	TermIDF       []float32
	TermOffsets   []uint32
	PostingsDoc   []uint32
	PostingsTF    []uint32
}

type BM25Scratch struct {
	Scores  []float32
	Touched []uint32
	Hits    []SearchHit
	LowBuf  []byte
}

func NewBM25Scratch(numDocs int) *BM25Scratch {
	return &BM25Scratch{
		Scores:  make([]float32, numDocs),
		Touched: make([]uint32, 0, 1024),
		Hits:    make([]SearchHit, 0, 1024),
		LowBuf:  make([]byte, 0, 256),
	}
}

func isWordByte(c byte) bool {
	if c >= 0x80 {
		return true
	}
	if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
		return true
	}
	return false
}

func lowerASCIIInto(src []byte, dst []byte) []byte {
	dst = dst[:0]
	for _, c := range src {
		if c >= 'A' && c <= 'Z' {
			c |= 0x20
		}
		dst = append(dst, c)
	}
	return dst
}

func LoadBM25Index(corpusPath string) (*BM25Index, error) {
	raw, err := os.ReadFile(corpusPath)
	if err != nil {
		return nil, err
	}
	rawLines := bytes.Split(raw, []byte{'\n'})
	// Drop trailing empty line from final newline
	if len(rawLines) > 0 && len(rawLines[len(rawLines)-1]) == 0 {
		rawLines = rawLines[:len(rawLines)-1]
	}

	numDocs := len(rawLines)
	idx := &BM25Index{
		NumDocs: uint32(numDocs),
		K1:      1.5,
		B:       0.75,
		DocLens: make([]uint32, numDocs),
		Vocab:   make(map[string]uint32, 2048),
	}

	var totalTokens uint64
	lowBuf := make([]byte, 0, 256)

	// Pass 1: build vocab & doc lengths
	for d, line := range rawLines {
		var dl uint32
		pos := 0
		n := len(line)
		for pos < n {
			if isWordByte(line[pos]) {
				start := pos
				pos++
				for pos < n && isWordByte(line[pos]) {
					pos++
				}
				norm := lowerASCIIInto(line[start:pos], lowBuf)
				if _, ok := idx.Vocab[string(norm)]; !ok {
					tid := uint32(len(idx.Vocab))
					idx.Vocab[string(norm)] = tid
				}
				dl++
			} else {
				pos++
			}
		}
		idx.DocLens[d] = dl
		totalTokens += uint64(dl)
	}

	V := len(idx.Vocab)
	if numDocs > 0 {
		idx.AvgDocLen = float32(float64(totalTokens) / float64(numDocs))
	}

	df := make([]uint32, V)
	lastDoc := make([]uint32, V)
	for i := range lastDoc {
		lastDoc[i] = math.MaxUint32
	}

	for d, line := range rawLines {
		pos := 0
		n := len(line)
		for pos < n {
			if isWordByte(line[pos]) {
				start := pos
				pos++
				for pos < n && isWordByte(line[pos]) {
					pos++
				}
				norm := lowerASCIIInto(line[start:pos], lowBuf)
				if tid, ok := idx.Vocab[string(norm)]; ok {
					if lastDoc[tid] != uint32(d) {
						lastDoc[tid] = uint32(d)
						df[tid]++
					}
				}
			} else {
				pos++
			}
		}
	}

	idx.TermOffsets = make([]uint32, V+1)
	idx.TermIDF = make([]float32, V)
	var totalPostings uint32
	N := float64(numDocs)
	for t := 0; t < V; t++ {
		idx.TermOffsets[t] = totalPostings
		totalPostings += df[t]
		nt := float64(df[t])
		idx.TermIDF[t] = float32(math.Log(1.0 + (N-nt+0.5)/(nt+0.5)))
	}
	idx.TermOffsets[V] = totalPostings

	idx.PostingsDoc = make([]uint32, totalPostings)
	idx.PostingsTF = make([]uint32, totalPostings)
	writeCur := make([]uint32, V)
	copy(writeCur, idx.TermOffsets[:V])
	for i := range lastDoc {
		lastDoc[i] = math.MaxUint32
	}

	// Pass 2: scatter doc-sorted postings
	for d, line := range rawLines {
		pos := 0
		n := len(line)
		for pos < n {
			if isWordByte(line[pos]) {
				start := pos
				pos++
				for pos < n && isWordByte(line[pos]) {
					pos++
				}
				norm := lowerASCIIInto(line[start:pos], lowBuf)
				if tid, ok := idx.Vocab[string(norm)]; ok {
					if lastDoc[tid] != uint32(d) {
						p := writeCur[tid]
						writeCur[tid]++
						idx.PostingsDoc[p] = uint32(d)
						idx.PostingsTF[p] = 1
						lastDoc[tid] = uint32(d)
					} else {
						idx.PostingsTF[writeCur[tid]-1]++
					}
				}
			} else {
				pos++
			}
		}
	}

	return idx, nil
}

// SearchIdiomatic uses standard Go per-request allocations (idiomatic style).
func (idx *BM25Index) SearchIdiomatic(query string, topK int) []SearchHit {
	if idx.NumDocs == 0 || topK <= 0 || len(query) == 0 {
		return nil
	}
	scores := make([]float32, idx.NumDocs)
	touched := make([]uint32, 0, 256)
	avgdl := idx.AvgDocLen
	if avgdl <= 0 {
		avgdl = 1.0
	}
	k1 := idx.K1
	b := idx.B

	qb := []byte(query)
	pos := 0
	n := len(qb)
	for pos < n {
		if isWordByte(qb[pos]) {
			start := pos
			pos++
			for pos < n && isWordByte(qb[pos]) {
				pos++
			}
			tok := lowerASCIIInto(qb[start:pos], nil)
			tid, ok := idx.Vocab[string(tok)]
			if !ok {
				continue
			}
			idf := idx.TermIDF[tid]
			p0 := idx.TermOffsets[tid]
			p1 := idx.TermOffsets[tid+1]
			for p := p0; p < p1; p++ {
				d := idx.PostingsDoc[p]
				tf := float32(idx.PostingsTF[p])
				dl := float32(idx.DocLens[d])
				denom := tf + k1*(1.0-b+b*(dl/avgdl))
				contrib := idf * ((tf * (k1 + 1.0)) / denom)
				if scores[d] == 0 && contrib > 0 {
					touched = append(touched, d)
				}
				scores[d] += contrib
			}
		} else {
			pos++
		}
	}

	if len(touched) == 0 {
		return nil
	}
	hits := make([]SearchHit, len(touched))
	for i, d := range touched {
		hits[i] = SearchHit{DocID: d, Score: scores[d]}
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].DocID < hits[j].DocID
	})
	if len(hits) > topK {
		hits = hits[:topK]
	}
	return hits
}

// SearchTuned reuses caller-provided scratch buffers (zero heap allocations per query).
func (idx *BM25Index) SearchTuned(query []byte, topK int, sc *BM25Scratch) []SearchHit {
	if idx.NumDocs == 0 || topK <= 0 || len(query) == 0 {
		return nil
	}
	scores := sc.Scores
	touched := sc.Touched[:0]
	avgdl := idx.AvgDocLen
	if avgdl <= 0 {
		avgdl = 1.0
	}
	k1 := idx.K1
	b := idx.B

	pos := 0
	n := len(query)
	for pos < n {
		if isWordByte(query[pos]) {
			start := pos
			pos++
			for pos < n && isWordByte(query[pos]) {
				pos++
			}
			sc.LowBuf = lowerASCIIInto(query[start:pos], sc.LowBuf)
			// Go compiler optimizes map lookup by string([]byte) with zero allocation
			tid, ok := idx.Vocab[string(sc.LowBuf)]
			if !ok {
				continue
			}
			idf := idx.TermIDF[tid]
			p0 := idx.TermOffsets[tid]
			p1 := idx.TermOffsets[tid+1]
			for p := p0; p < p1; p++ {
				d := idx.PostingsDoc[p]
				tf := float32(idx.PostingsTF[p])
				dl := float32(idx.DocLens[d])
				denom := tf + k1*(1.0-b+b*(dl/avgdl))
				contrib := idf * ((tf * (k1 + 1.0)) / denom)
				if scores[d] == 0 && contrib > 0 {
					touched = append(touched, d)
				}
				scores[d] += contrib
			}
		} else {
			pos++
		}
	}

	sc.Touched = touched
	if len(touched) == 0 {
		return nil
	}

	hits := sc.Hits[:0]
	for _, d := range touched {
		hits = append(hits, SearchHit{DocID: d, Score: scores[d]})
		scores[d] = 0 // O(touched) reset instead of O(num_docs) clear
	}
	sc.Hits = hits

	// Partial selection for small topK (much faster than full sort when touched >> topK)
	k := topK
	if k > len(hits) {
		k = len(hits)
	}
	for i := 0; i < k; i++ {
		best := i
		bestHit := hits[i]
		for j := i + 1; j < len(hits); j++ {
			h := hits[j]
			if h.Score > bestHit.Score || (h.Score == bestHit.Score && h.DocID < bestHit.DocID) {
				best = j
				bestHit = h
			}
		}
		if best != i {
			hits[i], hits[best] = hits[best], hits[i]
		}
	}
	return hits[:k]
}
