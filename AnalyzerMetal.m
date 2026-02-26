#import "AnalyzerMetal.h"
#import <Metal/Metal.h>



@interface AnalyzerMetal ()
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLBuffer> statesBuffer;
@property (nonatomic, strong) id<MTLBuffer> transitionsBuffer;
@property (nonatomic, strong) id<MTLBuffer> lemmaBuffer;
// Retained to keep the backing memory alive for the NoCopy buffers above
@property (nonatomic, strong) NSData *statesData;
@property (nonatomic, strong) NSData *transitionsData;
@property (nonatomic, strong) NSData *lemmasData;
@end

@implementation AnalyzerMetal

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if (self = [super init]) {
        _device = device;
        _commandQueue = [device newCommandQueue];

        NSError *error = nil;
        NSString *kernelSource = [NSString stringWithContentsOfFile:@"lookup_kernel.metal" encoding:NSUTF8StringEncoding error:&error];
        if (error) { NSLog(@" Failed to load Metal source: %@", error); return nil; }

        id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
        if (error) { NSLog(@" Failed to compile Metal: %@", error); return nil; }

        id<MTLFunction> kernel = [library newFunctionWithName:@"lookup_kernel"];
        _pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
        if (error) { NSLog(@" Failed to create pipeline: %@", error); return nil; }

        _statesData = [NSData dataWithContentsOfFile:@"resources/gpu_states.bin"];
        _transitionsData = [NSData dataWithContentsOfFile:@"resources/gpu_transitions.bin"];
        _lemmasData = [NSData dataWithContentsOfFile:@"resources/gpu_lemmas.bin"];

        _statesBuffer = [device newBufferWithBytesNoCopy:(void *)_statesData.bytes length:_statesData.length options:MTLResourceStorageModeShared deallocator:nil];
        _transitionsBuffer = [device newBufferWithBytesNoCopy:(void *)_transitionsData.bytes length:_transitionsData.length options:MTLResourceStorageModeShared deallocator:nil];
        _lemmaBuffer = [device newBufferWithBytesNoCopy:(void *)_lemmasData.bytes length:_lemmasData.length options:MTLResourceStorageModeShared deallocator:nil];
    }
    return self;
}



- (NSArray<NSString *> *)lemmatizeBatch:(NSArray<NSString *> *)words {
    NSUInteger total = words.count;
    NSMutableArray<NSString *> *allResults = [NSMutableArray arrayWithCapacity:total];
    for (NSUInteger i = 0; i < total; i++) {
        [allResults addObject:@""];
    }

    dispatch_group_t dispatchGroup = dispatch_group_create();
    dispatch_semaphore_t semaphore = dispatch_semaphore_create([[NSProcessInfo processInfo] activeProcessorCount]);

    NSUInteger BATCH_SIZE = 100000;

    for (NSUInteger i = 0; i < total; i += BATCH_SIZE) {
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        dispatch_group_enter(dispatchGroup);

        NSRange range = NSMakeRange(i, MIN(BATCH_SIZE, total - i));
        NSArray<NSString *> *batch = [words subarrayWithRange:range];

        NSUInteger max_word_len = 0;
        for (NSString *word in batch) {
            max_word_len = MAX(max_word_len, [word lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1);
        }

        NSUInteger bufferSize = batch.count * max_word_len;
        id<MTLBuffer> inputBuffer = [self.device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [self.device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> maxWordLenBuffer = [self.device newBufferWithBytes:&max_word_len length:sizeof(max_word_len) options:MTLResourceStorageModeShared];

        char *inputWords = (char *)inputBuffer.contents;
        memset(inputWords, 0, bufferSize);
        for (NSUInteger j = 0; j < batch.count; ++j) {
            const char *cstr = [batch[j] UTF8String];
            strncpy(inputWords + j * max_word_len, cstr, max_word_len - 1);
        }

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:self.pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:self.statesBuffer offset:0 atIndex:1];
        [encoder setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [encoder setBuffer:self.lemmaBuffer offset:0 atIndex:3];
        [encoder setBuffer:outputBuffer offset:0 atIndex:4];
        [encoder setBuffer:maxWordLenBuffer offset:0 atIndex:5];

        MTLSize gridSize = MTLSizeMake(batch.count, 1, 1);
        NSUInteger w = self.pipeline.threadExecutionWidth;
        MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            char *outputPtr = (char *)outputBuffer.contents;
            for (NSUInteger j = 0; j < batch.count; ++j) {
                NSString *s = [NSString stringWithUTF8String:outputPtr + j * max_word_len];
                @synchronized (allResults) {
                    allResults[range.location + j] = s ?: @"";
                }
            }
            dispatch_semaphore_signal(semaphore);
            dispatch_group_leave(dispatchGroup);
        }];

        [cmdBuf commit];
    }

    dispatch_group_wait(dispatchGroup, DISPATCH_TIME_FOREVER);

    return allResults;
}


@end
