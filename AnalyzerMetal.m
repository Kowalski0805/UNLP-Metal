#import "AnalyzerMetal.h"
#import <Metal/Metal.h>

#define MAX_WORD_LEN 37

@interface AnalyzerMetal ()
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLBuffer> statesBuffer;
@property (nonatomic, strong) id<MTLBuffer> transitionsBuffer;
@property (nonatomic, strong) id<MTLBuffer> lemmaBuffer;
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

        _statesBuffer = [self loadBufferFromFile:@"resources/states.bin"];
        _transitionsBuffer = [self loadBufferFromFile:@"resources/transitions.bin"];
        _lemmaBuffer = [self loadBufferFromFile:@"resources/lemmas.bin"];
    }
    return self;
}

- (id<MTLBuffer>)loadBufferFromFile:(NSString *)path {
    NSData *data = [NSData dataWithContentsOfFile:path];
    return [self.device newBufferWithBytes:data.bytes length:data.length options:0];
}

- (NSUInteger)calculateBatchSize {
    NSUInteger maxBytes = self.device.maxBufferLength;
    NSUInteger bytesPerWord = MAX_WORD_LEN * 2;
    NSUInteger estimated = maxBytes / bytesPerWord;
    NSUInteger cap = 100000;
    return MIN(estimated / 2, cap);
}

- (NSArray<NSString *> *)lemmatizeBatch:(NSArray<NSString *> *)words {
    NSUInteger BATCH_SIZE = [self calculateBatchSize];
    NSUInteger total = words.count;
    
    NSMutableArray<NSString *> *allResults = [NSMutableArray arrayWithCapacity:total];
    NSMutableArray<id<MTLBuffer>> *outputBuffers = [NSMutableArray array];
    NSMutableArray<id<MTLCommandBuffer>> *commandBuffers = [NSMutableArray array];
    NSMutableArray<NSArray<NSString *> *> *batches = [NSMutableArray array];

    for (NSUInteger i = 0; i < total; i += BATCH_SIZE) {
        NSUInteger length = MIN(BATCH_SIZE, total - i);
        NSArray<NSString *> *batch = [words subarrayWithRange:NSMakeRange(i, length)];
        [batches addObject:batch];

        NSUInteger bufferSize = length * MAX_WORD_LEN;
        char inputWords[length][MAX_WORD_LEN];
        memset(inputWords, 0, bufferSize);
        
        for (NSUInteger j = 0; j < length; ++j) {
            const char *cstr = [batch[j] UTF8String];
            strncpy(inputWords[j], cstr, MAX_WORD_LEN - 1);
        }

        id<MTLBuffer> inputBuffer = [self.device newBufferWithBytes:inputWords length:bufferSize options:0];
        id<MTLBuffer> outputBuffer = [self.device newBufferWithLength:bufferSize options:0];
        [outputBuffers addObject:outputBuffer];

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:self.pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:self.statesBuffer offset:0 atIndex:1];
        [encoder setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [encoder setBuffer:self.lemmaBuffer offset:0 atIndex:3];
        [encoder setBuffer:outputBuffer offset:0 atIndex:4];

        MTLSize gridSize = MTLSizeMake(length, 1, 1);
        NSUInteger w = self.pipeline.threadExecutionWidth;
        MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        
        [commandBuffers addObject:cmdBuf];
    }

    for (id<MTLCommandBuffer> buf in commandBuffers) {
        [buf waitUntilCompleted];
    }

    for (NSUInteger i = 0; i < batches.count; ++i) {
        NSArray<NSString *> *batch = batches[i];
        id<MTLBuffer> outputBuffer = outputBuffers[i];
        char *outputPtr = (char *)outputBuffer.contents;

        for (NSUInteger j = 0; j < batch.count; ++j) {
            NSString *s = [NSString stringWithUTF8String:outputPtr + j * MAX_WORD_LEN];
            [allResults addObject:s ?: @""];
        }
    }

    return allResults;
}


//- (NSArray<NSString *> *)lemmatizeBatch:(NSArray<NSString *> *)words {
//    NSUInteger BATCH_SIZE = [self calculateBatchSize];
//    NSMutableArray<NSString *> *allResults = [NSMutableArray array];
//    NSUInteger total = words.count;
//
//    for (NSUInteger i = 0; i < total; i += BATCH_SIZE) {
//        NSUInteger length = MIN(BATCH_SIZE, total - i);
//        NSArray<NSString *> *batch = [words subarrayWithRange:NSMakeRange(i, length)];
//        NSArray<NSString *> *result = [self lemmatizeWords:batch];
//        [allResults addObjectsFromArray:result];
//    }
//
//    return allResults;
//}
//
//- (NSArray<NSString *> *)lemmatizeWords:(NSArray<NSString *> *)words {
//    NSUInteger numWords = words.count;
//    if (numWords == 0) return @[];
//
//    NSUInteger bufferSize = numWords * MAX_WORD_LEN;
//    char *inputWords = calloc(numWords, MAX_WORD_LEN);
//    if (!inputWords) {
//        NSLog(@" Failed to allocate input buffer");
//        return @[];
//    }
//
//    for (NSUInteger i = 0; i < numWords; ++i) {
//        const char *cstr = [words[i] UTF8String];
//        strncpy(inputWords + i * MAX_WORD_LEN, cstr, MAX_WORD_LEN - 1);
//    }
//
//    id<MTLBuffer> inputBuffer = [self.device newBufferWithBytes:inputWords length:bufferSize options:0];
//    id<MTLBuffer> outputBuffer = [self.device newBufferWithLength:bufferSize options:0];
//    free(inputWords);
//
//    id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
//    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
//    [encoder setComputePipelineState:self.pipeline];
//    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
//    [encoder setBuffer:self.statesBuffer offset:0 atIndex:1];
//    [encoder setBuffer:self.transitionsBuffer offset:0 atIndex:2];
//    [encoder setBuffer:self.lemmaBuffer offset:0 atIndex:3];
//    [encoder setBuffer:outputBuffer offset:0 atIndex:4];
//
//    MTLSize gridSize = MTLSizeMake(numWords, 1, 1);
//    NSUInteger w = self.pipeline.threadExecutionWidth;
//    MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
//
//    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
//    [encoder endEncoding];
//    [cmdBuf commit];
//    [cmdBuf waitUntilCompleted];
//
//    NSMutableArray<NSString *> *results = [NSMutableArray arrayWithCapacity:numWords];
//    char *outputPtr = (char *)outputBuffer.contents;
//    for (NSUInteger i = 0; i < numWords; ++i) {
//        NSString *s = [NSString stringWithUTF8String:outputPtr + i * MAX_WORD_LEN];
//        [results addObject:s ?: @""];
//    }
//    return results;
//}

@end
