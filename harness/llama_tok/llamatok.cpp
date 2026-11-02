// llamatok: batch tokenizer over libllama, for differential testing.
//
// Usage: llamatok MODEL.gguf [--decode | --info] < cases > results
// With --decode, each input line is space-separated token ids and the output is the hex of
// llama_detokenize(ids, remove_special=false, unparse_special=true).
// With --info, no input is read; one JSON line describes the vocabulary's special-token
// behavior: add_bos/add_eos, the bos/eos/eot ids, every end-of-generation (EOG) token with
// its text and attribute bits, and the token ids of PROBE tokenized with and without
// add_special, which shows exactly what llama_tokenize puts around a prompt.
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

static std::string json_str(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);  // UTF-8 passes through; invalid bytes are fixed up by the reader
                }
        }
    }
    return out + "\"";
}

static std::string json_ids(const std::vector<llama_token> & ids) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); i++) {
        if (i) out += ",";
        out += std::to_string(ids[i]);
    }
    return out + "]";
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    std::vector<llama_token> toks(text.size() + 16);
    int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), add_special, true);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), add_special, true);
    }
    toks.resize(n < 0 ? 0 : n);
    return toks;
}

static const char * PROBE = "Hello world";

static int print_info(const llama_model * model, const llama_vocab * vocab) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    std::string eog = "[";
    bool first = true;
    for (llama_token t = 0; t < n; t++) {
        if (!llama_vocab_is_eog(vocab, t)) continue;
        if (!first) eog += ",";
        first = false;
        eog += "{\"id\":" + std::to_string(t) + ",\"text\":" + json_str(llama_vocab_get_text(vocab, t)) +
               ",\"attr\":" + std::to_string((int) llama_vocab_get_attr(vocab, t)) + "}";
    }
    eog += "]";
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::cout << "{\"n_tokens\":" << n
              << ",\"add_bos\":" << (llama_vocab_get_add_bos(vocab) ? "true" : "false")
              << ",\"add_eos\":" << (llama_vocab_get_add_eos(vocab) ? "true" : "false")
              << ",\"bos\":" << llama_vocab_bos(vocab)
              << ",\"eos\":" << llama_vocab_eos(vocab)
              << ",\"eot\":" << llama_vocab_eot(vocab)
              << ",\"eog\":" << eog
              << ",\"probe\":" << json_str(PROBE)
              << ",\"probe_plain\":" << json_ids(tokenize(vocab, PROBE, false))
              << ",\"probe_special\":" << json_ids(tokenize(vocab, PROBE, true))
              << ",\"chat_template\":" << (tmpl ? json_str(tmpl) : std::string("null"))
              << "}" << std::endl;
    return 0;
}

int main(int argc, char ** argv) {
    const std::string mode = argc == 3 ? argv[2] : "";
    if (argc < 2 || argc > 3 || (argc == 3 && mode != "--decode" && mode != "--info")) {
        fprintf(stderr, "usage: %s MODEL.gguf [--decode | --info] < lines\n", argv[0]);
        return 2;
    }
    const bool decode_only = mode == "--decode";
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
    if (mode == "--info") {
        const int rc = print_info(model, vocab);
        llama_model_free(model);
        llama_backend_free();
        return rc;
    }
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
