// llamatok: batch tokenizer over libllama, for differential testing.
//
// Usage: llamatok MODEL.gguf [--decode] < cases > results
// With --decode, each input line is space-separated token ids and the output is the hex of
// llama_detokenize(ids, remove_special=false, unparse_special=true).
// Input: one case per line, the UTF-8 text hex-encoded (so any byte, newline or NUL survives).
// Output: one line per case: space-separated token ids, a tab, and the hex of
// llama_detokenize(ids). Tokenization uses add_special=false, parse_special=true, which
// matches Hugging Face encode(text, add_special_tokens=False): added tokens in the text are
// matched, and no BOS/EOS is added.
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// llama.cpp's own log is silenced, but its warnings and errors are kept so a failed load can say why.
static std::string g_log;
static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        g_log += text;
        if (g_log.size() > 2000) g_log.erase(0, g_log.size() - 2000);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--decode")) {
        fprintf(stderr, "usage: %s MODEL.gguf [--decode] < lines\n", argv[0]);
        return 2;
    }
    const bool decode_only = argc == 3;
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        for (char & c : g_log) if (c == '\n') c = ' ';
        fprintf(stderr, "llama.cpp failed to load the model: %s\n", g_log.c_str());
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string line;
    std::vector<llama_token> toks;
    std::vector<char> buf;
    static const char * HEX = "0123456789abcdef";
    while (std::getline(std::cin, line)) {
        if (decode_only) {
            std::vector<llama_token> ids;
            size_t pos = 0;
            while (pos < line.size()) {
                size_t next = line.find(' ', pos);
                if (next == std::string::npos) next = line.size();
                if (next > pos) ids.push_back((llama_token) std::stol(line.substr(pos, next - pos)));
                pos = next + 1;
            }
            std::string out;
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            bool ok = true;
            for (llama_token t : ids) ok = ok && t >= 0 && t < n_vocab;
            if (!ok) {
                std::cout << "ERROR" << '\n' << std::flush;
                continue;
            }
            buf.resize(64 + 64 * ids.size());
            int32_t m = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), buf.data(), (int32_t) buf.size(), false, true);
            if (m < 0) {
                buf.resize(-m);
                m = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), buf.data(), (int32_t) buf.size(), false, true);
            }
            for (int32_t i = 0; i < m; i++) {
                unsigned char c = (unsigned char) buf[i];
                out.push_back(HEX[c >> 4]);
                out.push_back(HEX[c & 15]);
            }
            std::cout << out << '\n' << std::flush;
            continue;
        }
        std::string text;
        text.reserve(line.size() / 2);
        for (size_t i = 0; i + 1 < line.size(); i += 2) {
            int hi = hexval(line[i]), lo = hexval(line[i + 1]);
            if (hi < 0 || lo < 0) { fprintf(stderr, "bad hex\n"); return 1; }
            text.push_back((char) (hi * 16 + lo));
        }
        toks.resize(text.size() + 16);
        int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        }
        std::string out;
        for (int32_t i = 0; i < n; i++) {
            if (i) out.push_back(' ');
            out += std::to_string(toks[i]);
        }
        out.push_back('\t');
        if (n >= 0) {
            buf.resize(text.size() * 4 + 64);
            int32_t m = llama_detokenize(vocab, toks.data(), n, buf.data(), (int32_t) buf.size(), false, true);
            if (m < 0) {
                buf.resize(-m);
                m = llama_detokenize(vocab, toks.data(), n, buf.data(), (int32_t) buf.size(), false, true);
            }
            for (int32_t i = 0; i < m; i++) {
                unsigned char c = (unsigned char) buf[i];
                out.push_back(HEX[c >> 4]);
                out.push_back(HEX[c & 15]);
            }
        } else {
            out += "ERROR";
        }
        std::cout << out << '\n' << std::flush;
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
