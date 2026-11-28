package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"rcre_go_bench/internal/engine"
)

const maxBodyBytes = 1 << 20 // 1 MiB

type ServerApp struct {
	bm25       *engine.BM25Index
	gbdt       *engine.GBDTModel
	tuned      bool
	bufPool    sync.Pool
	bm25Pool   sync.Pool
	inferPool  sync.Pool
}

type inferScratch struct {
	flat  []float32
	preds []float32
}

func fnv1a64(b []byte) uint64 {
	var h uint64 = 0xcbf29ce484222325
	for _, c := range b {
		h ^= uint64(c)
		h *= 0x100000001b3
	}
	return h
}

func countLines(b []byte) int {
	if len(b) == 0 {
		return 0
	}
	n := bytes.Count(b, []byte{'\n'})
	if b[len(b)-1] == '\n' {
		return n
	}
	return n + 1
}

func writeJSONError(w http.ResponseWriter, code int, msg string) {
	body := `{"error":"` + msg + `"}`
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(code)
	_, _ = io.WriteString(w, body)
}

func (a *ServerApp) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Content-Length", "2")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("OK"))
}

// ---------- W2: POST /api/v1/validate ----------

type validateResp struct {
	Valid      bool   `json:"valid"`
	Bytes      int    `json:"bytes"`
	Codepoints int    `json:"codepoints"`
	Lines      int    `json:"lines"`
	FNV64      string `json:"fnv64"`
}

func (a *ServerApp) handleValidateIdiomatic(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	if r.ContentLength > maxBodyBytes {
		writeJSONError(w, http.StatusBadRequest, "bad_request")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil || len(body) > maxBodyBytes {
		writeJSONError(w, http.StatusBadRequest, "bad_request")
		return
	}

	valid := utf8.Valid(body)
	cp := 0
	if valid {
		cp = utf8.RuneCount(body)
	}
	resp := validateResp{
		Valid:      valid,
		Bytes:      len(body),
		Codepoints: cp,
		Lines:      countLines(body),
		FNV64:      fmt.Sprintf("%016x", fnv1a64(body)),
	}
	out, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}

func appendHex16(dst []byte, v uint64) []byte {
	const hex = "0123456789abcdef"
	var tmp [16]byte
	for i := 15; i >= 0; i-- {
		tmp[i] = hex[v&0xF]
		v >>= 4
	}
	return append(dst, tmp[:]...)
}

func (a *ServerApp) handleValidateTuned(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	if r.ContentLength > maxBodyBytes {
		writeJSONError(w, http.StatusBadRequest, "bad_request")
		return
	}
	rbPtr := a.bufPool.Get().(*[]byte)
	buf := (*rbPtr)[:0]
	defer func() {
		*rbPtr = buf
		a.bufPool.Put(rbPtr)
	}()

	if r.ContentLength >= 0 {
		cl := int(r.ContentLength)
		if cap(buf) < cl {
			buf = make([]byte, cl)
		} else {
			buf = buf[:cl]
		}
		if cl > 0 {
			if _, err := io.ReadFull(r.Body, buf); err != nil {
				writeJSONError(w, http.StatusBadRequest, "bad_request")
				return
			}
		}
	} else {
		var tmp [4096]byte
		for {
			n, err := r.Body.Read(tmp[:])
			if n > 0 {
				if len(buf)+n > maxBodyBytes {
					writeJSONError(w, http.StatusBadRequest, "bad_request")
					return
				}
				buf = append(buf, tmp[:n]...)
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				writeJSONError(w, http.StatusBadRequest, "bad_request")
				return
			}
		}
	}

	valid := utf8.Valid(buf)
	cp := 0
	if valid {
		cp = utf8.RuneCount(buf)
	}
	lines := countLines(buf)
	h := fnv1a64(buf)

	var outArr [160]byte
	out := outArr[:0]
	out = append(out, `{"valid":`...)
	if valid {
		out = append(out, "true"...)
	} else {
		out = append(out, "false"...)
	}
	out = append(out, `,"bytes":`...)
	out = strconv.AppendInt(out, int64(len(buf)), 10)
	out = append(out, `,"codepoints":`...)
	out = strconv.AppendInt(out, int64(cp), 10)
	out = append(out, `,"lines":`...)
	out = strconv.AppendInt(out, int64(lines), 10)
	out = append(out, `,"fnv64":"`...)
	out = appendHex16(out, h)
	out = append(out, `"}`...)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}

// ---------- W3: GET /api/v1/search?q=...&k=... ----------

func appendJSONString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			dst = append(dst, '\\', '"')
		case '\\':
			dst = append(dst, '\\', '\\')
		case '\n':
			dst = append(dst, '\\', 'n')
		case '\r':
			dst = append(dst, '\\', 'r')
		case '\t':
			dst = append(dst, '\\', 't')
		default:
			if c < 0x20 {
				dst = append(dst, '\\', 'u', '0', '0')
				const hex = "0123456789abcdef"
				dst = append(dst, hex[c>>4], hex[c&0xf])
			} else {
				dst = append(dst, c)
			}
		}
	}
	dst = append(dst, '"')
	return dst
}

func formatSearchJSON(out []byte, q string, hits []engine.SearchHit) []byte {
	out = append(out, `{"query":`...)
	out = appendJSONString(out, q)
	out = append(out, `,"count":`...)
	out = strconv.AppendInt(out, int64(len(hits)), 10)
	out = append(out, `,"hits":[`...)
	for i, h := range hits {
		if i > 0 {
			out = append(out, ',')
		}
		out = append(out, `{"doc_id":`...)
		out = strconv.AppendUint(out, uint64(h.DocID), 10)
		out = append(out, `,"score":`...)
		out = strconv.AppendFloat(out, float64(h.Score), 'f', 4, 32)
		out = append(out, '}')
	}
	out = append(out, `]}`...)
	return out
}

func (a *ServerApp) handleSearchIdiomatic(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	vals := r.URL.Query()
	q := vals.Get("q")
	k := 10
	if ks := vals.Get("k"); ks != "" {
		if parsed, err := strconv.Atoi(ks); err == nil {
			k = parsed
		}
	}
	if k < 1 {
		k = 1
	} else if k > 50 {
		k = 50
	}

	hits := a.bm25.SearchIdiomatic(q, k)
	out := formatSearchJSON(make([]byte, 0, 512), q, hits)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}

func parseSearchParamsFast(rawQuery string) (q string, k int) {
	k = 10
	for len(rawQuery) > 0 {
		var pair string
		if i := strings.IndexByte(rawQuery, '&'); i >= 0 {
			pair = rawQuery[:i]
			rawQuery = rawQuery[i+1:]
		} else {
			pair = rawQuery
			rawQuery = ""
		}
		eq := strings.IndexByte(pair, '=')
		if eq < 0 {
			continue
		}
		key, val := pair[:eq], pair[eq+1:]
		if key == "q" {
			if u, err := url.QueryUnescape(val); err == nil {
				q = u
			} else {
				q = val
			}
		} else if key == "k" {
			if parsed, err := strconv.Atoi(val); err == nil {
				k = parsed
			}
		}
	}
	if k < 1 {
		k = 1
	} else if k > 50 {
		k = 50
	}
	return q, k
}

func (a *ServerApp) handleSearchTuned(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	q, k := parseSearchParamsFast(r.URL.RawQuery)
	sc := a.bm25Pool.Get().(*engine.BM25Scratch)
	hits := a.bm25.SearchTuned([]byte(q), k, sc)

	var outArr [2048]byte
	out := formatSearchJSON(outArr[:0], q, hits)
	a.bm25Pool.Put(sc)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}

// ---------- W4: POST /api/v1/infer ----------

type inferReq struct {
	Rows [][]float32 `json:"rows"`
}

func formatInferJSON(out []byte, preds []float32) []byte {
	var sum float64
	for _, p := range preds {
		sum += float64(p)
	}
	mean := 0.0
	if len(preds) > 0 {
		mean = sum / float64(len(preds))
	}
	out = append(out, `{"count":`...)
	out = strconv.AppendInt(out, int64(len(preds)), 10)
	out = append(out, `,"mean":`...)
	out = strconv.AppendFloat(out, mean, 'f', 6, 64)
	out = append(out, `,"predictions":[`...)
	for i, p := range preds {
		if i > 0 {
			out = append(out, ',')
		}
		out = strconv.AppendFloat(out, float64(p), 'f', 4, 32)
	}
	out = append(out, `]}`...)
	return out
}

func (a *ServerApp) handleInferIdiomatic(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil || len(body) > maxBodyBytes {
		writeJSONError(w, http.StatusBadRequest, "bad_request")
		return
	}
	var req inferReq
	if err := json.Unmarshal(body, &req); err != nil || len(req.Rows) == 0 {
		writeJSONError(w, http.StatusBadRequest, "bad_request")
		return
	}
	D := a.gbdt.NumFeatures
	nRows := len(req.Rows)
	flat := make([]float32, nRows*D)
	for i, row := range req.Rows {
		if len(row) != D {
			writeJSONError(w, http.StatusBadRequest, "bad_request")
			return
		}
		copy(flat[i*D:(i+1)*D], row)
	}
	preds := make([]float32, nRows)
	a.gbdt.PredictFlat(flat, nRows, preds)

	out := formatInferJSON(make([]byte, 0, 1024), preds)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}

// parseInferFlatFast scans {"rows":[[f0,...,f15],...]} into dst with zero allocations.
func parseInferFlatFast(body []byte, D int, dst []float32) ([]float32, int, bool) {
	dst = dst[:0]
	idx := bytes.Index(body, []byte(`"rows"`))
	if idx < 0 {
		return nil, 0, false
	}
	i := idx + 6
	n := len(body)
	for i < n && body[i] != '[' {
		i++
	}
	if i >= n {
		return nil, 0, false
	}
	i++ // outer '['
	nRows := 0
	for i < n {
		for i < n && (body[i] == ' ' || body[i] == '\t' || body[i] == '\r' || body[i] == '\n' || body[i] == ',') {
			i++
		}
		if i >= n || body[i] == ']' {
			break
		}
		if body[i] != '[' {
			return nil, 0, false
		}
		i++ // inner '['
		cols := 0
		for i < n {
			for i < n && (body[i] == ' ' || body[i] == '\t' || body[i] == '\r' || body[i] == '\n' || body[i] == ',') {
				i++
			}
			if i >= n {
				return nil, 0, false
			}
			if body[i] == ']' {
				i++
				break
			}
			start := i
			for i < n && body[i] != ',' && body[i] != ']' && body[i] != ' ' && body[i] != '\t' && body[i] != '\r' && body[i] != '\n' {
				i++
			}
			v, err := strconv.ParseFloat(string(body[start:i]), 32)
			if err != nil {
				return nil, 0, false
			}
			dst = append(dst, float32(v))
			cols++
		}
		if cols != D {
			return nil, 0, false
		}
		nRows++
	}
	return dst, nRows, nRows > 0
}

func (a *ServerApp) handleInferTuned(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		return
	}
	if r.ContentLength > maxBodyBytes {
		writeJSONError(w, http.StatusBadRequest, "bad_request")
		return
	}
	rbPtr := a.bufPool.Get().(*[]byte)
	buf := (*rbPtr)[:0]
	defer func() {
		*rbPtr = buf
		a.bufPool.Put(rbPtr)
	}()

	if r.ContentLength >= 0 {
		cl := int(r.ContentLength)
		if cap(buf) < cl {
			buf = make([]byte, cl)
		} else {
			buf = buf[:cl]
		}
		if cl > 0 {
			if _, err := io.ReadFull(r.Body, buf); err != nil {
				writeJSONError(w, http.StatusBadRequest, "bad_request")
				return
			}
		}
	} else {
		var tmp [4096]byte
		for {
			n, err := r.Body.Read(tmp[:])
			if n > 0 {
				if len(buf)+n > maxBodyBytes {
					writeJSONError(w, http.StatusBadRequest, "bad_request")
					return
				}
				buf = append(buf, tmp[:n]...)
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				writeJSONError(w, http.StatusBadRequest, "bad_request")
				return
			}
		}
	}

	sc := a.inferPool.Get().(*inferScratch)
	flat, nRows, ok := parseInferFlatFast(buf, a.gbdt.NumFeatures, sc.flat)
	if !ok {
		a.inferPool.Put(sc)
		writeJSONError(w, http.StatusBadRequest, "bad_request")
		return
	}
	sc.flat = flat
	if cap(sc.preds) < nRows {
		sc.preds = make([]float32, nRows)
	} else {
		sc.preds = sc.preds[:nRows]
	}
	a.gbdt.PredictFlat(sc.flat, nRows, sc.preds)

	var outArr [2048]byte
	out := formatInferJSON(outArr[:0], sc.preds)
	a.inferPool.Put(sc)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}

func (a *ServerApp) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/health":
		a.handleHealth(w, r)
	case "/api/v1/validate":
		if a.tuned {
			a.handleValidateTuned(w, r)
		} else {
			a.handleValidateIdiomatic(w, r)
		}
	case "/api/v1/search":
		if a.tuned {
			a.handleSearchTuned(w, r)
		} else {
			a.handleSearchIdiomatic(w, r)
		}
	case "/api/v1/infer":
		if a.tuned {
			a.handleInferTuned(w, r)
		} else {
			a.handleInferIdiomatic(w, r)
		}
	default:
		writeJSONError(w, http.StatusNotFound, "not_found")
	}
}

func main() {
	host := flag.String("host", "127.0.0.1", "bind host")
	port := flag.Int("port", 18082, "bind port")
	workers := flag.Int("workers", 4, "GOMAXPROCS worker count")
	dataDir := flag.String("data", "../spec/data", "fixtures directory")
	mode := flag.String("mode", "idiomatic", "idiomatic or tuned")
	flag.Parse()

	if *workers > 0 {
		runtime.GOMAXPROCS(*workers)
	}

	t0 := time.Now()
	bm25, err := engine.LoadBM25Index(filepath.Join(*dataDir, "corpus_10k.txt"))
	if err != nil {
		log.Fatalf("load bm25: %v", err)
	}
	gbdt, err := engine.LoadGBDTModel(filepath.Join(*dataDir, "gbdt_model.bin"))
	if err != nil {
		log.Fatalf("load gbdt: %v", err)
	}

	app := &ServerApp{
		bm25:  bm25,
		gbdt:  gbdt,
		tuned: (*mode == "tuned"),
		bufPool: sync.Pool{
			New: func() any {
				b := make([]byte, 0, 16384)
				return &b
			},
		},
		bm25Pool: sync.Pool{
			New: func() any {
				return engine.NewBM25Scratch(int(bm25.NumDocs))
			},
		},
		inferPool: sync.Pool{
			New: func() any {
				return &inferScratch{
					flat:  make([]float32, 0, 1024),
					preds: make([]float32, 0, 64),
				}
			},
		},
	}

	srv := &http.Server{
		Addr:              fmt.Sprintf("%s:%d", *host, *port),
		Handler:           app,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	log.Printf("[go-%s] ready on %s (workers=%d, docs=%d, trees=%d, init=%.2fms)",
		*mode, srv.Addr, runtime.GOMAXPROCS(0), bm25.NumDocs, gbdt.NumTrees,
		float64(time.Since(t0).Microseconds())/1000.0)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}
