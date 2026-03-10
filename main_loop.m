#import <Foundation/Foundation.h>
#import "AnalyzerMetal.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s [--packed|--packed-col] <input_file.txt> [duration_s]\n", argv[0]);
            return 1;
        }

        BOOL usePacked    = (argc >= 3) && (strcmp(argv[1], "--packed") == 0);
        BOOL usePackedCol = (argc >= 3) && (strcmp(argv[1], "--packed-col") == 0);
        BOOL hasFlag = usePacked || usePackedCol;

        const char *filePath   = hasFlag ? argv[2] : argv[1];
        double runDuration     = 10.0;
        if (argc >= (hasFlag ? 4 : 3))
            runDuration = atof(hasFlag ? argv[3] : argv[2]);

        NSString *path = [NSString stringWithUTF8String:filePath];
        NSError *readError = nil;
        NSString *content = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&readError];
        if (readError || !content) {
            fprintf(stderr, "Failed to read file: %s\n", filePath);
            return 1;
        }

        NSArray<NSString *> *words = [content componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        words = [words filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"length > 0"]];
        if (words.count == 0) {
            fprintf(stderr, "No words in input file.\n");
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported on this device.\n");
            return 1;
        }

        AnalyzerMetal *analyzer = [[AnalyzerMetal alloc] initWithDevice:device];

        if (usePackedCol) {
            [analyzer benchLoopPackedColumn:words duration:runDuration];
        } else if (usePacked) {
            [analyzer benchLoopPacked:words duration:runDuration];
        } else {
            [analyzer benchLoopFixedStride:words duration:runDuration];
        }
    }
    return 0;
}
