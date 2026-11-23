/**
 * @file rcre_chat.c
 * @brief Local LLM CLI & Zero-Dependency OpenAI-Compatible HTTP Server (POSIX Sockets)
 */

#include "rcre.h"
#include "rcre_llm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#define RCRE_CHAT_POSIX 1
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

static void chat_print_usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --model <file.gguf>   Load GGUF v2/v3 LLaMA model (optional; builds Q8_0 demo model if omitted)\n"
           "  --cli                 Interactive terminal REPL\n"
           "  --serve <port>        Start OpenAI-compatible HTTP server (/v1/models, /v1/chat/completions)\n"
           "  --prompt <text>       Run single-shot generation\n"
           "  --max-tokens <n>      Max new tokens to sample (default: 16)\n"
           "  --max-requests <n>    Stop HTTP server after N requests (0 = infinite)\n"
           "  --threads <n>         ThreadPool workers for quantized GEMV\n"
           "  --demo                Run self-contained Q8_0 GGUF smoke generation\n",
           prog);
}

static void chat_escape_json(const char* src, char* dst, size_t max_dst) {
    size_t w = 0;
    for (size_t i = 0; src && src[i] != '\0' && w + 2 < max_dst; ++i) {
        char c = src[i];
        if (c == '"' || c == '\\') {
            dst[w++] = '\\';
            dst[w++] = c;
        } else if (c == '\n') {
            dst[w++] = '\\';
            dst[w++] = 'n';
        } else if ((unsigned char)c >= 32) {
            dst[w++] = c;
        }
    }
    dst[w] = '\0';
}

static void chat_run_completion(
    const RCRE_LlamaModel* model,
    RCRE_LlamaState* state,
    const RCRE_ByteBPE* bpe,
    RCRE_Sampler* sampler,
    const char* prompt,
    size_t max_new,
    char* out_text,
    size_t max_out
) {
    int32_t prompt_ids[256];
    size_t n_prompt = rcre_byte_bpe_encode(bpe, prompt, strlen(prompt), prompt_ids, 256);
    if (n_prompt == 0) {
        prompt_ids[0] = (int32_t)bpe->bos_id;
        n_prompt = 1;
    }
    for (size_t i = 0; i < n_prompt; ++i) {
        if ((size_t)prompt_ids[i] >= model->cfg.vocab_size) {
            prompt_ids[i] %= (int32_t)model->cfg.vocab_size;
        }
    }

    int32_t gen_ids[256];
    if (max_new > 256) max_new = 256;
    size_t gen_n = 0;
    if (rcre_llama_generate(model, state, sampler, prompt_ids, n_prompt, gen_ids, max_new, &gen_n) != RCRE_OK) {
        snprintf(out_text, max_out, "(generation error)");
        return;
    }
    (void)rcre_byte_bpe_decode(bpe, gen_ids, gen_n, out_text, max_out);
}

static int chat_serve_http(
    int port,
    int max_requests,
    const RCRE_LlamaModel* model,
    RCRE_LlamaState* state,
    const RCRE_ByteBPE* bpe,
    RCRE_Sampler* sampler,
    size_t max_tokens
) {
#if !defined(RCRE_CHAT_POSIX)
    (void)port; (void)max_requests; (void)model; (void)state; (void)bpe; (void)sampler; (void)max_tokens;
    fprintf(stderr, "HTTP server requires POSIX sockets.\n");
    return 1;
#else
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return 1;
    int yes = 1;
    (void)setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons((uint16_t)port);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) != 0 || listen(srv, 16) != 0) {
        close(srv);
        return 1;
    }

    printf("🚀 RCRE OpenAI-compatible Server listening on http://127.0.0.1:%d\n", port);
    fflush(stdout);

    int served = 0;
    while (max_requests <= 0 || served < max_requests) {
        int cli = accept(srv, nullptr, nullptr);
        if (cli < 0) break;
        served++;

        char req[8192];
        ssize_t n = recv(cli, req, sizeof(req) - 1U, 0);
        if (n > 0) {
            req[n] = '\0';
            char resp_body[4096];
            if (strncmp(req, "GET /v1/models", 14) == 0) {
                snprintf(resp_body, sizeof(resp_body),
                         "{\"object\":\"list\",\"data\":[{\"id\":\"rcre-llama-local\",\"object\":\"model\"}]}");
            } else {
                const char* prompt = "hello";
                const char* body = strstr(req, "\r\n\r\n");
                char extracted[512] = "hello";
                if (body) {
                    const char* c = strstr(body, "\"content\"");
                    if (c) {
                        const char* q1 = strchr(c + 9, '"');
                        const char* q2 = q1 ? strchr(q1 + 1, '"') : nullptr;
                        if (q1 && q2 && q2 > q1 + 1) {
                            size_t len = (size_t)(q2 - q1 - 1);
                            if (len >= sizeof(extracted)) len = sizeof(extracted) - 1U;
                            memcpy(extracted, q1 + 1, len);
                            extracted[len] = '\0';
                            prompt = extracted;
                        }
                    }
                }
                char raw_out[1024] = {0};
                char esc_out[2048] = {0};
                chat_run_completion(model, state, bpe, sampler, prompt, max_tokens, raw_out, sizeof(raw_out));
                chat_escape_json(raw_out, esc_out, sizeof(esc_out));
                snprintf(resp_body, sizeof(resp_body),
                         "{\"id\":\"chatcmpl-rcre\",\"object\":\"chat.completion\","
                         "\"model\":\"rcre-llama-local\",\"choices\":[{\"index\":0,"
                         "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                         "\"finish_reason\":\"length\"}]}", esc_out);
            }

            char hdr[256];
            int hlen = snprintf(hdr, sizeof(hdr),
                                "HTTP/1.1 200 OK\r\n"
                                "Content-Type: application/json\r\n"
                                "Content-Length: %zu\r\n"
                                "Connection: close\r\n\r\n",
                                strlen(resp_body));
            (void)send(cli, hdr, (size_t)hlen, 0);
            (void)send(cli, resp_body, strlen(resp_body), 0);
        }
        close(cli);
    }
    close(srv);
    return 0;
#endif
}

int main(int argc, char** argv) {
    rcre_hal_init();

    const char* model_path = nullptr;
    const char* prompt = nullptr;
    bool interactive = false;
    int serve_port = 0;
    int max_requests = 0;
    size_t max_tokens = 16;
    size_t n_threads = 2;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            chat_print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--cli") == 0) {
            interactive = true;
        } else if (strcmp(argv[i], "--demo") == 0) {
            prompt = "RCRE runs everywhere";
        } else if (strcmp(argv[i], "--serve") == 0 && i + 1 < argc) {
            serve_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-requests") == 0 && i + 1 < argc) {
            max_requests = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = (size_t)atoi(argv[++i]);
        }
    }

    int err = RCRE_OK;
    RCRE_LlamaModel* model = nullptr;
    if (model_path) {
        model = rcre_llama_load_gguf(model_path, &err);
        if (!model) {
            fprintf(stderr, "Failed to load GGUF '%s' (err=%d)\n", model_path, err);
            return 1;
        }
    } else {
        RCRE_LlamaConfig cfg = {
            .vocab_size  = 320,
            .dim         = 64,
            .hidden_dim  = 128,
            .n_layers    = 2,
            .n_heads     = 4,
            .n_kv_heads  = 2, /* Grouped-Query Attention (2:1 GQA) */
            .max_seq_len = 128,
            .rms_eps     = 1e-5f,
            .rope_theta  = 10000.0f
        };
        model = rcre_llama_create(cfg, RCRE_LLM_W_Q8_0, 42ULL);
    }

    RCRE_ThreadPool* pool = rcre_threadpool_create_ex(n_threads, RCRE_QOS_PERF);
    model->pool = pool;
    RCRE_LlamaState* state = rcre_llama_state_create(&model->cfg);
    RCRE_ByteBPE* bpe = rcre_byte_bpe_create(model->cfg.vocab_size > 258 ? model->cfg.vocab_size : 258);
    RCRE_Sampler* sampler = rcre_sampler_create(model->cfg.vocab_size, (RCRE_SamplerConfig){
        .temperature    = 0.8f,
        .top_k          = 32,
        .top_p          = 0.9f,
        .min_p          = 0.05f,
        .repeat_penalty = 1.1f,
        .repeat_window  = 16,
        .rng_state      = 1337ULL
    });

    int rc = 0;
    if (serve_port > 0) {
        rc = chat_serve_http(serve_port, max_requests, model, state, bpe, sampler, max_tokens);
    } else if (interactive) {
        printf("RCRE Chat CLI (dim=%zu, layers=%zu, kv_heads=%zu). Type prompt or 'quit':\n",
               model->cfg.dim, model->cfg.n_layers, model->cfg.n_kv_heads);
        char line[1024];
        char out[1024];
        while (printf("> "), fflush(stdout), fgets(line, sizeof(line), stdin)) {
            line[strcspn(line, "\r\n")] = '\0';
            if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;
            if (line[0] == '\0') continue;
            chat_run_completion(model, state, bpe, sampler, line, max_tokens, out, sizeof(out));
            printf("Assistant: %s\n", out);
        }
    } else {
        const char* p = prompt ? prompt : "Hello RCRE";
        char out[1024];
        chat_run_completion(model, state, bpe, sampler, p, max_tokens, out, sizeof(out));
        printf("Prompt: %s\nCompletion (%zu tokens): %s\n", p, max_tokens, out);
    }

    rcre_sampler_free(sampler);
    rcre_byte_bpe_free(bpe);
    rcre_llama_state_free(state);
    rcre_threadpool_destroy(pool);
    rcre_llama_free(model);
    return rc;
}
