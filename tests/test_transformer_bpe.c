/**
 * @file test_transformer_bpe.c
 * @brief Subword Byte-Pair Encoding & Character Tokenizer Verification (ISO C23)
 */

#include "rcre.h"
#include "test_framework.h"

static void test_char_tokenizer(void) {
    RCRE_CharTokenizer tok;
    rcre_char_tokenizer_init(&tok);
    ASSERT_INT_EQ(0, tok.vocab_size);

    const char *corpus = "data/tinystories.txt";
    bool ok = rcre_char_tokenizer_train_from_file(&tok, corpus);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tok.vocab_size > 20);

    const char *test_str = "Hello world! 123";
    int tokens[64];
    int n = rcre_char_tokenizer_encode(&tok, test_str, strlen(test_str), tokens, 64);
    ASSERT_TRUE(n > 0);

    char decoded[64];
    size_t dec_len = rcre_char_tokenizer_decode(&tok, tokens, n, decoded, sizeof(decoded));
    ASSERT_UINT_EQ((size_t)n, dec_len);
    ASSERT_INT_EQ(0, strcmp(test_str, decoded));

    /* Serialization test */
    const char *tok_file = "data/temp_char_tok.bin";
    ASSERT_TRUE(rcre_char_tokenizer_save(&tok, tok_file));

    RCRE_CharTokenizer loaded_tok;
    ASSERT_TRUE(rcre_char_tokenizer_load(&loaded_tok, tok_file));
    ASSERT_INT_EQ(tok.vocab_size, loaded_tok.vocab_size);

    char redec[64];
    rcre_char_tokenizer_decode(&loaded_tok, tokens, n, redec, sizeof(redec));
    ASSERT_INT_EQ(0, strcmp(test_str, redec));

    remove(tok_file);
}

static void test_bpe_tokenizer(void) {
    const char *corpus = "data/tinystories.txt";
    int target_vocab = 384;

    RCRE_BPETokenizer *tok = rcre_bpe_create();
    ASSERT_NOT_NULL(tok);

    bool trained = rcre_bpe_train_from_file(tok, corpus, target_vocab);
    ASSERT_TRUE(trained);
    ASSERT_TRUE(tok->vocab_size > 256);
    ASSERT_TRUE(tok->num_rules > 0);

    /* Serialization test */
    const char *bpe_file = "data/temp_bpe.bin";
    ASSERT_TRUE(rcre_bpe_save(tok, bpe_file));

    RCRE_BPETokenizer *loaded_tok = rcre_bpe_create();
    ASSERT_NOT_NULL(loaded_tok);
    ASSERT_TRUE(rcre_bpe_load(loaded_tok, bpe_file));
    ASSERT_INT_EQ(tok->vocab_size, loaded_tok->vocab_size);
    ASSERT_INT_EQ(tok->num_rules, loaded_tok->num_rules);

    /* Roundtrip and compression test */
    const char *sentence = "Once upon a time, there was a little girl who loved to play with her little dog.";
    size_t raw_len = strlen(sentence);

    int encoded[256];
    int n_tokens = rcre_bpe_encode(loaded_tok, sentence, raw_len, encoded, 256);
    ASSERT_TRUE(n_tokens > 0);
    ASSERT_TRUE((size_t)n_tokens < raw_len); /* Subword compression verified */

    char decoded[512];
    size_t dec_len = rcre_bpe_decode(loaded_tok, encoded, n_tokens, decoded, sizeof(decoded));
    ASSERT_UINT_EQ(raw_len, dec_len);
    ASSERT_INT_EQ(0, strcmp(sentence, decoded));

    rcre_bpe_free(tok);
    rcre_bpe_free(loaded_tok);
    remove(bpe_file);
}

int main(void) {
    test_suite_begin("RCRE Byte-Pair Encoding (BPE) & Character Tokenizers");
    TEST_RUN(test_char_tokenizer);
    TEST_RUN(test_bpe_tokenizer);
    return test_suite_report();
}
