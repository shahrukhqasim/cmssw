#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include <math.h>

#include "RecoLocalCalo/HGCalRecAlgos/interface/GPUHist2D.h"


namespace BinnerGPU {


  __global__ void kernel_compute_histogram(RechitForBinning*dInputData, histogram2D<int,ETA_BINS, PHI_BINS, MAX_DEPTH> *dOutputData, const size_t numRechits) {

    size_t rechitLocation = blockIdx.x * blockDim.x + threadIdx.x;

    if(rechitLocation >= numRechits)
        return;

    float eta = dInputData[rechitLocation].eta;
    float phi = dInputData[rechitLocation].phi;
    unsigned int index = dInputData[rechitLocation].index;
   
    dOutputData->fillBinGPU(eta, phi, index);

  }


  float minEta = 1.6;
  float maxEta = 3.0;
  float minPhi = -M_PI;
  float maxPhi = M_PI;

  std::shared_ptr<int> computeBins(std::vector<RechitForBinning> layerData) {
    histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH> hOutputData(minEta, maxEta, minPhi, maxPhi);

    // Allocate memory and put data into device
    histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH> *dOutputData;
    RechitForBinning* dInputData;
    cudaMalloc(&dOutputData, sizeof(histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH>));
    cudaMalloc(&dInputData, sizeof(RechitForBinning)*layerData.size());
    cudaMemcpy(dInputData, layerData.data(), sizeof(RechitForBinning)*layerData.size(), cudaMemcpyHostToDevice);
    cudaMemset(dOutputData, 0x00, sizeof(histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH>));
    cudaMemcpy(dOutputData, &hOutputData, sizeof(histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH>), cudaMemcpyHostToDevice);
  
    // Call the kernel
    const dim3 blockSize(1024,1,1);
    const dim3 gridSize(ceil(layerData.size()/1024.0),1,1);
    kernel_compute_histogram <<<gridSize,blockSize>>>(dInputData, dOutputData, layerData.size());

    // Copy result back!
    cudaMemcpy(dOutputData, &hOutputData, sizeof(histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH>), cudaMemcpyDeviceToHost);

    // Free all the memory
    cudaFree(dOutputData);
    cudaFree(dInputData);

    
    return hOutputData;
  }

}
