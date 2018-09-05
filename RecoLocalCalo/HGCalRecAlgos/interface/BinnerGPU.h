#include <memory>
#include <iostream>
#include <vector>

#ifndef Binner_GPU_h
#define Binner_GPU_h

struct RechitForBinning {
        unsigned int index;
        float eta;
        float phi;
};


typedef std::vector<std::vector<RechitForBinning>> BinningData;
typedef std::vector<RechitForBinning> LayerData;


namespace BinnerGPU {
    // eta_width = 0.05
    // phi_width = 0.05
    // 2*pi/0.05 = 125
    // 1.4/0.05 = 28
    // 20 (as heuristic)

const int ETA_BINS=28;
const int PHI_BINS=126;
const int MAX_DEPTH=20;
const int MAX_NEIGHBORS=20;



std::shared_ptr<int> computeBins(LayerData layerData);
std::shared_ptr<int> computeBins(BinningData caloData);


}


#endif
