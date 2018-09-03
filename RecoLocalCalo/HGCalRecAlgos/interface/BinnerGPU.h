#include <memory>
#include <iostream>
#include <vector>


#ifndef Binner_GPU_h
#define Binner_GPU_h



struct RechitForBinning {
        long index;
        float eta;
        float phi;
};

typedef std::vector<std::vector<RechitForBinning>> BinningData;
typedef std::vector<RechitForBinning> LayerData;

std::shared_ptr<long> computeBins(LayerData layerData);

#endif
