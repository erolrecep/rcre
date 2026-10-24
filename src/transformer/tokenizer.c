/**
 * @file tokenizer.c
 * @brief RCRE Character & Byte-Pair Encoding (BPE) Subword Tokenizers (ISO C23)
 */

#include "transformer_internal.h"

/* ========================================================================= */
/* Character-Level Tokenizer                                                 */
/* ========================================================================= */

void rcre_char_tokenizer_init(RCRE_CharTokenizer *tok) {
    if (!tok) return;
    tok->vocab_size = 0;
    for (int i = 0; i < 256; i++) {
        tok->char_to_id[i] = -1;
        tok->id_to_char[i] = 0;
    }
}

bool rcre_char_tokenizer_train_from_file(RCRE_CharTokenizer *tok, const char *filepath) {
    rcre_char_tokenizer_init(tok);

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        return false;
    }

    bool seen[256] = {false};
    int c;
    while ((c = fgetc(f)) != EOF) {
        seen[(unsigned char)c] = true;
    }
    fclose(f);

    int id = 0;
    for (int i = 0; i < 256; i++) {
        if (seen[i]) {
            tok->char_to_id[i] = id;
            tok->id_to_char[id] = (unsigned char)i;
            id++;
        }
    }
    tok->vocab_size = id;
    return true;
}

bool rcre_char_tokenizer_save(const RCRE_CharTokenizer *tok, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;
    if (fwrite(&tok->vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(tok->id_to_char, sizeof(unsigned char), (size_t)tok->vocab_size, f) != (size_t)tok->vocab_size) {
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

bool rcre_char_tokenizer_load(RCRE_CharTokenizer *tok, const char *filepath) {
    rcre_char_tokenizer_init(tok);
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;
    if (fread(&tok->vocab_size, sizeof(int), 1, f) != 1) {
        fclose(f);
        return false;
    }
    if (fread(tok->id_to_char, sizeof(unsigned char), (size_t)tok->vocab_size, f) != (size_t)tok->vocab_size) {
        fclose(f);
        return false;
    }
    fclose(f);

    for (int i = 0; i < tok->vocab_size; i++) {
        unsigned char ch = tok->id_to_char[i];
        tok->char_to_id[ch] = i;
    }
    return true;
}

int rcre_char_tokenizer_encode(const RCRE_CharTokenizer *tok, const char *text, size_t len, int *out_tokens) {
    int count = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)text[i];
        int id = tok->char_to_id[ch];
        if (id != -1) {
            out_tokens[count++] = id;
        }
    }
    return count;
}

size_t rcre_char_tokenizer_decode(const RCRE_CharTokenizer *tok, const int *tokens, size_t num_tokens, char *out_text, size_t max_out_len) {
    size_t i = 0;
    for (; i < num_tokens && i + 1 < max_out_len; i++) {
        int token_id = tokens[i];
        if (token_id >= 0 && token_id < tok->vocab_size) {
            out_text[i] = (char)tok->id_to_char[token_id];
        } else {
            out_text[i] = '?';
        }
    }
    out_text[i] = '\0';
    return i;
}

void rcre_char_tokenizer_print_info(const RCRE_CharTokenizer *tok) {
    printf("========================================\n");
    printf("   Tokenizer Vocabulary Info (V = %d)   \n", tok->vocab_size);
    printf("========================================\n");
    for (int i = 0; i < tok->vocab_size; i++) {
        unsigned char c = tok->id_to_char[i];
        if (c == '\n') {
            printf("[%2d]: '\\n'  ", i);
        } else if (c == ' ') {
            printf("[%2d]: ' '   ", i);
        } else if (c == '\t') {
            printf("[%2d]: '\\t'  ", i);
        } else if (c >= 32 && c <= 126) {
            printf("[%2d]: '%c'   ", i, c);
        } else {
            printf("[%2d]: 0x%02X ", i, c);
        }
        if ((i + 1) % 6 == 0) printf("\n");
    }
    printf("\n========================================\n");
}

/* ========================================================================= */
/* Byte-Pair Encoding (BPE) Subword Tokenizer                                */
/* ========================================================================= */

RCRE_BPETokenizer* rcre_bpe_create(void) {
    RCRE_BPETokenizer *tok = (RCRE_BPETokenizer*)malloc(sizeof(RCRE_BPETokenizer));
    if (!tok) return nullptr;

    tok->vocab_size = 256;
    tok->num_rules = 0;
    tok->rules = nullptr;

    tok->id_to_token_str = (char**)malloc(sizeof(char*) * 2048);
    for (int i = 0; i < 256; i++) {
        tok->id_to_token_str[i] = (char*)malloc(2);
        tok->id_to_token_str[i][0] = (char)(unsigned char)i;
        tok->id_to_token_str[i][1] = '\0';
    }
    for (int i = 256; i < 2048; i++) {
        tok->id_to_token_str[i] = nullptr;
    }
    return tok;
}

bool rcre_bpe_train_from_file(RCRE_BPETokenizer *tok, const char *filepath, int target_vocab_size) {
    if (target_vocab_size <= 256 || target_vocab_size > 2048) {
        return false;
    }

    FILE *f = fopen(filepath, "rb");
    if (!f) return false;

    size_t max_sample = 250000;
    unsigned char *raw_bytes = (unsigned char*)malloc(max_sample);
    if (!raw_bytes) { fclose(f); return false; }
    size_t n_read = fread(raw_bytes, 1, max_sample, f);
    fclose(f);

    if (n_read < 100) {
        free(raw_bytes);
        return false;
    }

    size_t current_len = n_read;
    uint16_t *tokens = (uint16_t*)malloc(sizeof(uint16_t) * current_len);
    if (!tokens) { free(raw_bytes); return false; }
    for (size_t i = 0; i < current_len; i++) {
        tokens[i] = (uint16_t)raw_bytes[i];
    }
    free(raw_bytes);

    int max_merges = target_vocab_size - 256;
    tok->rules = (RCRE_BPEMergeRule*)malloc(sizeof(RCRE_BPEMergeRule) * max_merges);
    tok->num_rules = 0;

    int current_vocab = 256;

    for (int m = 0; m < max_merges; m++) {
        size_t map_size = (size_t)current_vocab * current_vocab;
        uint32_t *counts = (uint32_t*)calloc(map_size, sizeof(uint32_t));
        if (!counts) break;

        for (size_t i = 0; i + 1 < current_len; i++) {
            uint16_t t1 = tokens[i];
            uint16_t t2 = tokens[i + 1];
            counts[t1 * current_vocab + t2]++;
        }

        uint32_t best_count = 0;
        uint16_t best_p1 = 0, best_p2 = 0;
        for (int p1 = 0; p1 < current_vocab; p1++) {
            for (int p2 = 0; p2 < current_vocab; p2++) {
                uint32_t c = counts[p1 * current_vocab + p2];
                if (c > best_count) {
                    best_count = c;
                    best_p1 = (uint16_t)p1;
                    best_p2 = (uint16_t)p2;
                }
            }
        }
        free(counts);

        if (best_count <= 1) {
            break;
        }

        uint16_t new_id = (uint16_t)current_vocab;
        tok->rules[tok->num_rules].id1 = best_p1;
        tok->rules[tok->num_rules].id2 = best_p2;
        tok->rules[tok->num_rules].new_id = new_id;
        tok->num_rules++;

        size_t len1 = strlen(tok->id_to_token_str[best_p1]);
        size_t len2 = strlen(tok->id_to_token_str[best_p2]);
        char *merged_str = (char*)malloc(len1 + len2 + 1);
        memcpy(merged_str, tok->id_to_token_str[best_p1], len1);
        memcpy(merged_str + len1, tok->id_to_token_str[best_p2], len2);
        merged_str[len1 + len2] = '\0';
        tok->id_to_token_str[new_id] = merged_str;

        size_t write_idx = 0;
        for (size_t i = 0; i < current_len; i++) {
            if (i + 1 < current_len && tokens[i] == best_p1 && tokens[i + 1] == best_p2) {
                tokens[write_idx++] = new_id;
                i++;
            } else {
                tokens[write_idx++] = tokens[i];
            }
        }
        current_len = write_idx;
        current_vocab++;
    }

    tok->vocab_size = current_vocab;
    free(tokens);
    return true;
}

bool rcre_bpe_save(const RCRE_BPETokenizer *tok, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;
    fwrite(&tok->vocab_size, sizeof(int), 1, f);
    fwrite(&tok->num_rules, sizeof(int), 1, f);
    fwrite(tok->rules, sizeof(RCRE_BPEMergeRule), tok->num_rules, f);

    for (int i = 0; i < tok->vocab_size; i++) {
        size_t len = strlen(tok->id_to_token_str[i]);
        fwrite(&len, sizeof(size_t), 1, f);
        fwrite(tok->id_to_token_str[i], 1, len, f);
    }
    fclose(f);
    return true;
}

bool rcre_bpe_load(RCRE_BPETokenizer *tok, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;

    if (fread(&tok->vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (fread(&tok->num_rules, sizeof(int), 1, f) != 1) { fclose(f); return false; }

    tok->rules = (RCRE_BPEMergeRule*)malloc(sizeof(RCRE_BPEMergeRule) * tok->num_rules);
    if (fread(tok->rules, sizeof(RCRE_BPEMergeRule), tok->num_rules, f) != (size_t)tok->num_rules) {
        fclose(f);
        return false;
    }

    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->id_to_token_str[i]) free(tok->id_to_token_str[i]);
        size_t len = 0;
        if (fread(&len, sizeof(size_t), 1, f) != 1) { fclose(f); return false; }
        tok->id_to_token_str[i] = (char*)malloc(len + 1);
        if (fread(tok->id_to_token_str[i], 1, len, f) != len) { fclose(f); return false; }
        tok->id_to_token_str[i][len] = '\0';
    }
    fclose(f);
    return true;
}

int rcre_bpe_encode(const RCRE_BPETokenizer *tok, const char *text, size_t text_len, int *out_tokens, int max_tokens) {
    if (text_len == 0) return 0;

    size_t cur_len = text_len;
    uint16_t *buf = (uint16_t*)malloc(sizeof(uint16_t) * cur_len);
    if (!buf) return 0;
    for (size_t i = 0; i < cur_len; i++) {
        buf[i] = (unsigned char)text[i];
    }

    for (int r = 0; r < tok->num_rules; r++) {
        uint16_t p1 = tok->rules[r].id1;
        uint16_t p2 = tok->rules[r].id2;
        uint16_t new_id = tok->rules[r].new_id;

        size_t write_idx = 0;
        for (size_t i = 0; i < cur_len; i++) {
            if (i + 1 < cur_len && buf[i] == p1 && buf[i + 1] == p2) {
                buf[write_idx++] = new_id;
                i++;
            } else {
                buf[write_idx++] = buf[i];
            }
        }
        cur_len = write_idx;
    }

    int count = (cur_len < (size_t)max_tokens) ? (int)cur_len : max_tokens;
    for (int i = 0; i < count; i++) {
        out_tokens[i] = (int)buf[i];
    }
    free(buf);
    return count;
}

size_t rcre_bpe_decode(const RCRE_BPETokenizer *tok, const int *tokens, size_t num_tokens, char *out_text, size_t max_out_len) {
    size_t written = 0;
    out_text[0] = '\0';

    for (size_t i = 0; i < num_tokens; i++) {
        int token_id = tokens[i];
        if (token_id >= 0 && token_id < tok->vocab_size) {
            const char *piece = tok->id_to_token_str[token_id];
            size_t plen = strlen(piece);
            if (written + plen < max_out_len) {
                memcpy(out_text + written, piece, plen);
                written += plen;
                out_text[written] = '\0';
            } else {
                break;
            }
        }
    }
    return written;
}

void rcre_bpe_print_summary(const RCRE_BPETokenizer *tok) {
    printf("===================================================================\n");
    printf("   Byte-Pair Encoding (BPE) Summary: V = %d (%d Merges Learned)   \n",
           tok->vocab_size, tok->num_rules);
    printf("===================================================================\n");
    printf("Top 15 Most Frequent Merged Tokens:\n");
    for (int i = 0; i < 15 && i < tok->num_rules; i++) {
        int id = 256 + i;
        printf("  Token [%d]: \"", id);
        for (const char *p = tok->id_to_token_str[id]; *p; p++) {
            if (*p == '\n') printf("\\n");
            else if (*p == '\t') printf("\\t");
            else if (*p == ' ') printf(" ");
            else printf("%c", *p);
        }
        printf("\"\n");
    }
    printf("===================================================================\n");
}

void rcre_bpe_free(RCRE_BPETokenizer *tok) {
    if (tok) {
        if (tok->rules) free(tok->rules);
        if (tok->id_to_token_str) {
            for (int i = 0; i < tok->vocab_size; i++) {
                if (tok->id_to_token_str[i]) free(tok->id_to_token_str[i]);
            }
            free(tok->id_to_token_str);
        }
        free(tok);
    }
}
