#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"

std::shared_ptr<long> computeBins(std::vector<RechitForBinning> layerData) {
    std::shared_ptr<long> hostData(new long[ETA_BINS*PHI_BINS*MAX_DEPTH]);

    // TODO: Do the computation here
    
    return hostData;
}