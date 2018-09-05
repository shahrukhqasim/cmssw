#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>



namespace BinnerGPU {

typedef GPU::VecArray<int,MAX_DEPTH> RequiredBin;


__global__ void kernel_compute_histogram(RechitForBinning*dInputData, RequiredBin*dOutputData, const size_t numRechits) {

    size_t rechitLocation = blockIdx.x * blockDim.x + threadIdx.x;

    if(rechitLocation >= numRechits)
        return;

    float eta = dInputData[rechitLocation].eta;
    float phi = dInputData[rechitLocation].phi;
    unsigned int index = dInputData[rechitLocation].index;

    int etaIndex = floor((abs(eta) - 1.6) / 0.05);
    int phiIndex = floor((phi + M_PI) / 0.05);

    dOutputData[phiIndex*ETA_BINS + etaIndex].push_back(index);
}



std::shared_ptr<int> computeBins(std::vector<RechitForBinning> layerData) {
    std::shared_ptr<int> hOutputData(new int[ETA_BINS*PHI_BINS*MAX_DEPTH]);

    // Allocate memory and put data into device
    RequiredBin* dOutputData;
    RechitForBinning* dInputData;
    cudaMalloc(&dOutputData, sizeof(RequiredBin)*ETA_BINS*PHI_BINS);
    cudaMalloc(&dInputData, sizeof(RechitForBinning)*layerData.size());
    cudaMemcpy(dInputData, layerData.data(), sizeof(RechitForBinning)*layerData.size(), cudaMemcpyHostToDevice);
    cudaMemset(dOutputData, 0x00, sizeof(RequiredBin)*ETA_BINS*PHI_BINS);

    // Call the kernel
    const dim3 blockSize(1024,1,1);
    const dim3 gridSize(ceil(layerData.size()/1024.0),1,1);
    kernel_compute_histogram <<<gridSize,blockSize>>>(dInputData, dOutputData, layerData.size());

    // Copy result back!
    cudaMemcpy(dOutputData, hOutputData.get(), sizeof(int)*ETA_BINS*PHI_BINS*MAX_DEPTH, cudaMemcpyDeviceToHost);

    // Free all the memory
    cudaFree(dOutputData);
    cudaFree(dInputData);

    
    return hOutputData;
}





__global__ void kernel_search(RechitForBinning*dInputData, RequiredBin*dOutputData, const size_t numRechits, float delta_c) {

    size_t rechitLocation = blockIdx.x * blockDim.x + threadIdx.x;

    size_t rechitLocation1 = rechitLocation / 1024;
    size_t rechitLocation2 = rechitLocation % 1024;

    if(rechitLocation1 >= numRechits or rechitLocation2 >= numRechits)
        return;

    float x1 = dInputData[rechitLocation1].eta;
    float y1 = dInputData[rechitLocation1].phi;

    float x2 = dInputData[rechitLocation2].eta;
    float y2 = dInputData[rechitLocation2].phi;

    if (std::abs(x1-x2) < delta_c && std::abs(y1-y2) < delta_c) {
        dOutputData[rechitLocation1].push_back(dInputData[rechitLocation2].index);
    }
}




std::shared_ptr<int> searchThing(std::vector<RechitForBinning> layerData, float delta_c) {
    std::shared_ptr<int> hOutputData(new int[layerData.size()*MAX_NEIGHBORS]);

    // Allocate memory and put data into device
    RequiredBin* dOutputData;
    RechitForBinning* dInputData;
    cudaMalloc(&dOutputData, sizeof(RequiredBin)*MAX_NEIGHBORS);
    cudaMalloc(&dInputData, sizeof(RechitForBinning)*layerData.size());
    cudaMemcpy(dInputData, layerData.data(), sizeof(RechitForBinning)*layerData.size(), cudaMemcpyHostToDevice);
    cudaMemset(dOutputData, 0x00, sizeof(RequiredBin)*MAX_NEIGHBORS);

    // Call the kernel
    const dim3 blockSize(1024,1,1);
    const dim3 gridSize(ceil((layerData.size()*layerData.size())/1024.0),1,1);
    kernel_search <<<gridSize,blockSize>>>(dInputData, dOutputData, layerData.size(), delta_c);

    // Copy result back!
    cudaMemcpy(dOutputData, hOutputData.get(), sizeof(int)*MAX_NEIGHBORS*MAX_DEPTH, cudaMemcpyDeviceToHost);

    // Free all the memory
    cudaFree(dOutputData);
    cudaFree(dInputData);

    
    return hOutputData;
}

}