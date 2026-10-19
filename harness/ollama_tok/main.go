// Command ollamatok encodes and decodes strings with Ollama's MLX-runner tokenizer
// (github.com/ollama/ollama/mlxrunner/tokenizer, vendored unmodified in ./tokenizer).
//
// Usage: ollamatok TOKENIZER_JSON [-decode] < cases.jsonl > results.jsonl
// Each input line is {"id": ..., "text": ...}; each output line is
// {"id": ..., "ids": [...], "decoded": "...", "error": "..."}.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"infercheck/ollamatok/tokenizer"
)

type inCase struct {
	ID   string  `json:"id"`
	Text string  `json:"text"`
	IDs  []int32 `json:"ids"` // decode mode: decode these ids instead of encoding text
}

type outCase struct {
	ID      string  `json:"id"`
	IDs     []int32 `json:"ids"`
	Decoded string  `json:"decoded"`
	Error   string  `json:"error,omitempty"`
}

func main() {
	if len(os.Args) < 2 || len(os.Args) > 3 || (len(os.Args) == 3 && os.Args[2] != "-decode") {
		fmt.Fprintln(os.Stderr, "usage: ollamatok TOKENIZER_JSON [-decode] < cases.jsonl")
		os.Exit(2)
	}
	decodeOnly := len(os.Args) == 3
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	tok, err := tokenizer.LoadFromBytes(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(3)
	}
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1<<20), 1<<26)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)
	for in.Scan() {
		var c inCase
		if err := json.Unmarshal(in.Bytes(), &c); err != nil {
			fmt.Fprintln(os.Stderr, "bad input:", err)
			os.Exit(1)
		}
		res := outCase{ID: c.ID}
		func() {
			defer func() {
				if r := recover(); r != nil {
					res.Error = fmt.Sprint("panic: ", r)
				}
			}()
			if decodeOnly {
				res.IDs = c.IDs
				res.Decoded = tok.Decode(c.IDs)
				return
			}
			res.IDs = tok.Encode(c.Text, false)
			if res.IDs == nil {
				res.IDs = []int32{}
			}
			res.Decoded = tok.Decode(res.IDs)
		}()
		if err := enc.Encode(res); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		// flush per line so a caller can hold the process open and query it one case at a time
		if err := out.Flush(); err != nil {
			os.Exit(1)
		}
	}
	if err := in.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
