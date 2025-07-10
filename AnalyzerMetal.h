#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface AnalyzerMetal : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (NSArray<NSString *> *)lemmatizeBatch:(NSArray<NSString *> *)words;

@end
