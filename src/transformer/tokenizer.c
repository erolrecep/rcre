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
    if (tok->vocab_size < 0 || tok->vocab_size > 256) return false;
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;
    const uint32_t magic = 0x31434352u; /* "RCC1" */
    const uint8_t version = RCRE_CHAR_TOK_FORMAT_VERSION;
    bool ok = (fwrite(&magic, sizeof(magic), 1, f) == 1) &&
              (fwrite(&version, sizeof(version), 1, f) == 1) &&
              (fwrite(&tok->vocab_size, sizeof(int), 1, f) == 1) &&
              (fwrite(tok->id_to_char, sizeof(unsigned char), (size_t)tok->vocab_size, f) == (size_t)tok->vocab_size);
    fclose(f);
    return ok;
}

bool rcre_char_tokenizer_load(RCRE_CharTokenizer *tok, const char *filepath) {
    /* id_to_char is a fixed 256-entry member: the file's vocab_size is
     * validated against it BEFORE any byte is read into the array. */
    rcre_char_tokenizer_init(tok);
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;
    uint32_t magic = 0;
    uint8_t version = 0;
    int vocab_size = 0;
    bool ok = (fread(&magic, sizeof(magic), 1, f) == 1) &&
              (fread(&version, sizeof(version), 1, f) == 1) &&
              (fread(&vocab_size, sizeof(int), 1, f) == 1);
    if (ok) {
        ok = (magic == 0x31434352u) && (version == RCRE_CHAR_TOK_FORMAT_VERSION) &&
             (vocab_size >= 0) && (vocab_size <= 256);
    }
    char buf[256];
    if (ok) {
        ok = (fread(buf, sizeof(unsigned char), (size_t)vocab_size, f) == (size_t)vocab_size);
    }
    fclose(f);
    if (!ok) return false;
    for (int i = 0; i < vocab_size; i++) {
        unsigned char ch = (unsigned char)buf[i];
        tok->id_to_char[i] = (char)ch;
        tok->char_to_id[ch] = i;
    }
    tok->vocab_size = vocab_size;
    return true;
}

int rcre_char_tokenizer_encode(const RCRE_CharTokenizer *tok, const char *text, size_t len, int *out_tokens, int max_tokens) {
    int count = 0;
    for (size_t i = 0; i < len && count < max_tokens; i++) {
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
    /* Retrain: release previously merged strings. The 256 base byte strings
     * (entries 0..255) are owned by rcre_bpe_create() and survive. */
    if (tok->rules) { free(tok->rules); tok->rules = NULL; }
    for (int i = 256; i < tok->vocab_size; i++) { free(tok->id_to_token_str[i]); tok->id_to_token_str[i] = nullptr; }
    tok->rules = (RCRE_BPEMergeRule*)malloc(sizeof(RCRE_BPEMergeRule) * (size_t)max_merges);
    tok->num_rules = 0;
    if (!tok->rules) { free(tokens); return false; }

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
    if (!tok->id_to_token_str) return false;
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;
    const uint32_t magic = 0x31455052u; /* "RPE1" */
    const uint8_t version = RCRE_BPE_FORMAT_VERSION;
    bool ok = (fwrite(&magic, sizeof(magic), 1, f) == 1) &&
              (fwrite(&version, sizeof(version), 1, f) == 1) &&
              (fwrite(&tok->vocab_size, sizeof(int), 1, f) == 1) &&
              (fwrite(&tok->num_rules, sizeof(int), 1, f) == 1) &&
              (tok->num_rules == 0 ||
               fwrite(tok->rules, sizeof(RCRE_BPEMergeRule), (size_t)tok->num_rules, f) == (size_t)tok->num_rules);
    for (int i = 0; ok && i < tok->vocab_size; i++) {
        const char *str = tok->id_to_token_str[i];
        size_t len = str ? strlen(str) : 0;
        ok = (fwrite(&len, sizeof(size_t), 1, f) == 1);
        if (ok && len > 0) ok = (fwrite(str, 1, len, f) == len);
    }
    fclose(f);
    return ok;
}

bool rcre_bpe_load(RCRE_BPETokenizer *tok, const char *filepath) {
    /* Requires a bpe_create()d tokenizer (id_to_token_str is sized for the
     * 2048-vocab ceiling). Rejects unknown magic/version and out-of-range
     * sizes; a trained tokenizer always has vocab_size == 256 + num_rules.
     * Nothing is committed to `tok` unless the whole file is valid. */
    if (!tok || !tok->id_to_token_str) return false;
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;

    uint32_t magic = 0;
    uint8_t version = 0;
    int vocab_size = 0;
    int num_rules = 0;
    bool ok = (fread(&magic, sizeof(magic), 1, f) == 1) &&
              (fread(&version, sizeof(version), 1, f) == 1) &&
              (fread(&vocab_size, sizeof(int), 1, f) == 1) &&
              (fread(&num_rules, sizeof(int), 1, f) == 1);
    if (ok) {
        ok = (magic == 0x31455052u) && (version == RCRE_BPE_FORMAT_VERSION) &&
             (vocab_size > 256) && (vocab_size <= 2048) &&
             (num_rules >= 0) && (num_rules == vocab_size - 256);
    }

    RCRE_BPEMergeRule *rules = NULL;
    char **strings = NULL;
    if (ok) {
        if (num_rules > 0) {
            rules = (RCRE_BPEMergeRule*)malloc(sizeof(RCRE_BPEMergeRule) * (size_t)num_rules);
            ok = rules != NULL &&
                 (fread(rules, sizeof(RCRE_BPEMergeRule), (size_t)num_rules, f) == (size_t)num_rules);
            if (!ok) { free(rules); rules = NULL; }
        }
    }
    if (ok) {
        strings = (char**)calloc((size_t)vocab_size, sizeof(char*));
        ok = (strings != NULL);
        for (int i = 0; ok && i < vocab_size; i++) {
            size_t len = 0;
            ok = (fread(&len, sizeof(size_t), 1, f) == 1) && (len <= 512);
            if (!ok) break;
            char *str = (char*)malloc(len + 1);
            ok = (str != NULL);
            if (ok && len > 0) ok = (fread(str, 1, len, f) == len);
            if (!ok) { free(str); break; }
            str[len] = '\0';
            strings[i] = str;
        }
        if (!ok) {
            for (int i = 0; i < vocab_size; i++) free(strings[i]);
            free(strings);
            free(rules);
            strings = NULL;
            rules = NULL;
        }
    }
    fclose(f);
    if (!ok) return false;

    /* Commit: base strings 0..255 stay create()d; swap the merged range. */
    if (tok->rules) free(tok->rules);
    for (int i = 256; i < tok->vocab_size; i++) { free(tok->id_to_token_str[i]); tok->id_to_token_str[i] = nullptr; }
    tok->rules = rules;
    tok->num_rules = num_rules;
    tok->vocab_size = vocab_size;
    for (int i = 256; i < vocab_size; i++) tok->id_to_token_str[i] = strings[i];
    for (int i = 0; i < 256; i++) free(strings[i]); /* base copies read from file */
    free(strings);
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
            /* Full scan: entries beyond vocab_size are expected to be
             * nullptr, but freeing non-NULL stragglers keeps failed edge
             * cases leak-free. */
            for (int i = 0; i < 2048; i++) {
                if (tok->id_to_token_str[i]) free(tok->id_to_token_str[i]);
            }
            free(tok->id_to_token_str);
        }
        free(tok);
    }
}
