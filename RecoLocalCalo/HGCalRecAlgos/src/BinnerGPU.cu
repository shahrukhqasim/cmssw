#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"


namespace BinnerGPU {

typedef GPU::VecArray<long,MAX_DEPTH> RequiredBin;


std::shared_ptr<long> computeBins(std::vector<RechitForBinning> layerData) {
    std::shared_ptr<long> hostData(new long[ETA_BINS*PHI_BINS*MAX_DEPTH]);

    // TODO: Do the computation here

    
    return hostData;
}

}