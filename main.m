#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import "AnalyzerMetal.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <input_file.txt>\n", argv[0]);
            return 1;
        }

        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSError *readError = nil;
        NSString *content = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&readError];

        if (readError || !content) {
            fprintf(stderr, "Failed to read file: %s\n", argv[1]);
            return 1;
        }
         
        NSArray<NSString *> *words = [content componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        words = [words filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"length > 0"]];

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported on this device.\n");
            return 1;
        }

        AnalyzerMetal *analyzer = [[AnalyzerMetal alloc] initWithDevice:device];

        uint64_t start = mach_absolute_time();
        NSArray<NSString *> *results = [analyzer lemmatizeBatch:words];
        uint64_t end = mach_absolute_time();

        mach_timebase_info_data_t timeInfo;
        mach_timebase_info(&timeInfo);
        uint64_t elapsedNano = (end - start) * timeInfo.numer / timeInfo.denom;
        double elapsedMs = elapsedNano / 1e6;

        for (NSUInteger i = 0; i < results.count; i++) {
            printf("%s → %s\n", [words[i] UTF8String], [results[i] UTF8String]);
        }

        printf("GPU time: %.3f ms for %lu words\n", elapsedMs, (unsigned long)results.count);
    }
    return 0;
}

