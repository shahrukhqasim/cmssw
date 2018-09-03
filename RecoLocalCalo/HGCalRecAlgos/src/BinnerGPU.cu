#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"

// eta_width = 0.05
// phi_width = 0.05
// 2*pi/0.05 = 125
// 1.4/0.05 = 28
// 20 (as heuristic)

#define ETA_BINS 28
#define PHI_BINS 125
#define MAX_DEPTH 20


std::shared_ptr<long> computeBins(std::vector<RechitForBinning> layerData) {
    std::shared_ptr<long> hostData(new long[ETA_BINS*PHI_BINS*MAX_DEPTH]);

    // TODO: Do the computation here
    
    return hostData;
}