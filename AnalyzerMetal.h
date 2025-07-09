#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

//#define MAX_WORD_LEN 37

typedef struct {
    uint32_t transition_start_idx;
    uint32_t num_transitions;
    int32_t lemma_offset;
} GpuState;

typedef struct {
    char c;
    uint32_t next_state;
} GpuTransition;

@interface AnalyzerMetal : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (NSArray<NSString *> *)lemmatizeWords:(NSArray<NSString *> *)words;
@end
