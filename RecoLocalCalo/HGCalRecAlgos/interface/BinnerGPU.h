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

// eta_width = 0.05
// phi_width = 0.05
// 2*pi/0.05 = 125
// 1.4/0.05 = 28
// 20 (as heuristic)

std::shared_ptr<long> computeBins(std::vector<RechitForBinning> layerData);

#endif
