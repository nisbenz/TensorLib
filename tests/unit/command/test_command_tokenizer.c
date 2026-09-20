#include <assert.h>
#include <string.h>

#include "command_tokenizer.h"

int main(void)
{
    static const unsigned char text[] =
        "REQUEST: show the current status\nCOMMAND: TOOL git ACTION status\n";
    command_tokenizer tokenizer;
    uint16_t encoded[128];
    unsigned char decoded[sizeof(text) + 1];
    size_t encoded_length = 0;
    size_t decoded_length = 0;

    command_tokenizer_init(&tokenizer);
    assert(command_tokenizer_train(&tokenizer, text, sizeof(text) - 1, 512) == 0);
    assert(command_tokenizer_encode(&tokenizer, text, sizeof(text) - 1,
                                    encoded, 128, &encoded_length) == 0);
    assert(encoded_length > 0);
    assert(command_tokenizer_decode(&tokenizer, encoded, encoded_length,
                                    decoded, sizeof(decoded), &decoded_length) == 0);
    assert(decoded_length == sizeof(text) - 1);
    assert(memcmp(decoded, text, decoded_length) == 0);
    command_tokenizer_destroy(&tokenizer);
    return 0;
}
