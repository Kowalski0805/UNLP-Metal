#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

#define MAX_WORD_LEN 37

typedef struct {
    uint32_t transition_start_idx;
    uint32_t num_transitions;
    int32_t lemma_offset;
} GpuState;

typedef struct {
    char c;
    uint32_t next_state;
} GpuTransition;

NSData *loadFile(NSString *path) {
    return [NSData dataWithContentsOfFile:path];
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            NSLog(@"Usage: ./trie_cpu_test word1 word2 ...");
            return 1;
        }

        NSData *statesData = loadFile(@"resources/states.bin");
        NSData *transData = loadFile(@"resources/transitions.bin");
        NSData *lemmasData = loadFile(@"resources/lemmas.bin");

        const GpuState *states = (const GpuState *)statesData.bytes;
        const GpuTransition *transitions = (const GpuTransition *)transData.bytes;
        const char *lemmas = (const char *)lemmasData.bytes;

        uint64_t start = mach_absolute_time();

        for (int i = 1; i < argc; ++i) {
            const char *word = argv[i];
            int state = 0;
            int matched = 1;

            for (int j = 0; j < MAX_WORD_LEN && word[j] != '\0'; ++j) {
                char ch = word[j];
                const GpuState s = states[state];
                int found = 0;

                for (uint32_t k = 0; k < s.num_transitions; ++k) {
                    GpuTransition t = transitions[s.transition_start_idx + k];
                    if (t.c == ch) {
                        state = t.next_state;
                        found = 1;
                        break;
                    }
                }

                if (!found) {
                    matched = 0;
                    break;
                }
            }

            if (matched && states[state].lemma_offset >= 0) {
                const char *lemma = lemmas + states[state].lemma_offset;
                printf("%s → %s\n", word, lemma);
            } else {
                printf("%s → not found in trie\n", word);
            }
        }

        uint64_t end = mach_absolute_time();

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        uint64_t elapsed_ns = (end - start) * info.numer / info.denom;
        printf("CPU time: %.3f ms\n", elapsed_ns / 1e6);
    }
    return 0;
}

