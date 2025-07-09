//
//  main.m
//
//
//  Created by Vitalii Apostolyuk on 2025-07-08.
//
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import "AnalyzerMetal.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal not supported");
            return -1;
        }

        AnalyzerMetal *analyzer = [[AnalyzerMetal alloc] initWithDevice:device];
        if (!analyzer) {
            NSLog(@"Failed to initialize AnalyzerMetal");
            return -1;
        }

        NSMutableArray<NSString *> *inputs = [NSMutableArray array];
        NSString *filePath = [NSString stringWithUTF8String:argv[1]];
        NSString *content = [NSString stringWithContentsOfFile:filePath encoding:NSUTF8StringEncoding error:nil];
        NSArray<NSString *> *lines = [content componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
        [inputs addObjectsFromArray:lines];

        
        uint64_t start = mach_absolute_time();
        NSArray<NSString *> *results = [analyzer lemmatizeWords:inputs];
        uint64_t end = mach_absolute_time();

        
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        uint64_t elapsed_ns = (end - start) * info.numer / info.denom;
        double elapsed_ms = elapsed_ns / 1e6;

        
        for (NSUInteger i = 0; i < results.count; i++) {
            printf("%s → %s\n", [inputs[i] UTF8String], [results[i] UTF8String]);
        }

        printf("⏱ GPU time: %.3f ms\n", elapsed_ms);
    }
    return 0;
}
