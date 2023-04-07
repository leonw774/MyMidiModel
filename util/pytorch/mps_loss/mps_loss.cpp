#include <torch/extension.h>

#include <vector>
#include <utility>
#include <limits>
#include <iostream>

using namespace torch::indexing;

/*
For each position of predicted label, find the target label that minimizes the cross entropy loss in its MPS

Input:

    batchedPredLogitsList is a list of Float64 tensors
        - length = out_attr_num
        - each element is a tenors with shape of (batch_size, seq_size, attr_vocabs_size)

    batchedTarget is a Int64 tensors with shape of (batch_size, seq_size, out_attr_num)

    batchedMPSIndices is a list of Int32 lists
        - length = batch_size
        - each element is a list of Int32 with variable size

    We use at::tensor because we don't need them to be differentiable (no_grad)

Output:

    The modified target labels that minimizes the loss in each position's MPS
*/

// Using triu_indices to process data "in parallel"
// Comparing to the loop version of the function, this method have 4% increase in performance
// Tested on two RTX3090 with huggingface/accelerate default configuration
at::Tensor findMinPermuLossTarget_TriU(
    std::vector<at::Tensor> batchedPredLogitsList,
    at::Tensor batchedTarget,
    std::vector<std::vector<int32_t>> batchedMPSIndices
) {
    int64_t outAttrNum = batchedTarget.size(2);
    for (int64_t batchNum = 0; batchNum < batchedTarget.size(0); batchNum++) {
        // std::cout << "batchNum " << batchNum << std::endl;
        std::vector<int32_t> curMPSIndices= batchedMPSIndices[batchNum]; // copy the vector

        // Flatten MPS with size >= 2
        std::vector<at::Tensor> flattenMPSTargetList;
        std::vector<std::vector<at::Tensor>> flattenMPSPredLogitList;
        for (int k = 0; k < outAttrNum; k++) {
            flattenMPSPredLogitList.push_back(std::vector<at::Tensor>());
        }
        // std::cout << "curMPSIndices.size() " << curMPSIndices.size() << std::endl;
        for (size_t mpsNum = 0; mpsNum < curMPSIndices.size() - 1; mpsNum++) {
            int beginIndex = curMPSIndices[mpsNum];
            // if (beginIndex < 0) {
            //     continue;
            // }
            int endIndex = curMPSIndices[mpsNum+1];
            int mpsSize = endIndex - beginIndex;
            // if (mpsSize < 0) {
            //     throw std::range_error("Negtive mpsSize");
            // }
            if (mpsSize == 2) {
                flattenMPSTargetList.push_back(
                    // batchedTarget[batchNum, beginIndex:endIndex]
                    batchedTarget.index({batchNum, Slice(beginIndex, endIndex)})
                );
                for (int k = 0; k < outAttrNum; k++) {
                    flattenMPSPredLogitList[k].push_back(
                        // batchedPredLogitsList[k][batch_number, begin_index].expand((2, -1))
                        batchedPredLogitsList[k].index({batchNum, beginIndex}).expand({2, -1})
                    );
                }
            }
            else if (mpsSize > 2) {
                // triu_indices returns [ [0, 0, 0, ..., N-2, N-2, N-1], [0, 1, 2, ..., N-2, N-1, N-1] ]
                // curMPSTriu = triu_indices(mpsSize, mpsSize, 0)[:, :-1] (drop last)
                int curFlattenMPSLength = mpsSize * (mpsSize + 1) / 2 - 1;
                at::Tensor curMPSTriu = at::triu_indices(mpsSize, mpsSize, 0).index({
                    Ellipsis, Slice(None, curFlattenMPSLength)
                });
                // curMPSTarget = batchedTarget[batch_number, begin_index:end_index]
                at::Tensor curMPSTarget = batchedTarget.index({batchNum, Slice(beginIndex, endIndex)});
                // flattenCurMPSTarget = curMPSTarget.unsqueeze(0).expand({mpsSize, -1, -1}))
                // exapnd dim to (1, mps_size, out_attr_number) then repeat to (mps_size, mps_size, out_attr_number)
                // t1, t2, t3 ... -> t1, t2, t3 ...
                //                   t1, t2, t3 ...
                //                   :
                //                   t1, t2, t3 ...
                // flattenCurMPSTarget = flattenCurMPSTarget[curMPSTriu[0], curMPSTriu[1]].flatten(end_dim=-2)
                // t1, t2, t3 ... tn, t2, t3, t4 ... tn, ... , tn-1, tn
                at::Tensor flattenCurMPSTarget =
                    (curMPSTarget.unsqueeze(0).expand({mpsSize, -1, -1})).index({
                        curMPSTriu.index({0}), curMPSTriu.index({1})
                    }).flatten(0LL, -2LL);
                flattenMPSTargetList.push_back(flattenCurMPSTarget);

                for (int k = 0; k < outAttrNum; k++) {
                    at::Tensor curMPSPredK = batchedPredLogitsList[k].index({batchNum, Slice(beginIndex, endIndex)});
                    // flattenCurMPSPredK = curMPSPredK.unsqueeze(1).expand((-1, mps_size, -1))
                    // exapnd dim to (mps_size, 1, out_attr_number) then repeat to (mps_size, mps_size, attr_vocabs_size)
                    // l1        l1, l1, l1 ...
                    // l2  -->   l2, l2, l2 ...
                    // l3        l3, l3, l3 ...
                    // :         :
                    // flattenCurMPSPredK = flattenCurMPSPredK[curMPSTriu[0], curMPSTriu[1]].flatten(end_dim=-2)
                    // l1, l1, l1, ... , l1 (n times), l2, l2, l2, ... , l2 (n-1 times), ... , ln-1, ln-1 (2 times)
                    at::Tensor flattenCurMPSPredK =
                        curMPSPredK.unsqueeze(1).expand({-1, mpsSize, -1}).index({
                            curMPSTriu.index({0}), curMPSTriu.index({1})
                        }).flatten(0LL, -2LL);
                    flattenMPSPredLogitList[k].push_back(flattenCurMPSPredK);
                }
            }
        }
        // std::cout << "flattenMPSTargetList.size() " << flattenMPSTargetList.size() << std::endl;
        // for (auto flattenMPSTarget: flattenMPSTargetList) {
        //     std::cout << " " << flattenMPSTarget.size(0);
        // }
        // std::cout << std::endl;
        // Calculate cross entropys
        if (flattenMPSTargetList.size() == 0) {
            continue;
        }
        // catFlattenMPSTarget = torch.cat(flattenMPSTargetList, dim=0)
        at::Tensor catFlattenMPSTarget = at::concat(flattenMPSTargetList, 0);
        // std::cout << "catFlattenMPSTarget.size(0) " << catFlattenMPSTarget.size(0) << std::endl;
        // std::cout << "catFlattenMPSTarget:\n" << catFlattenMPSTarget << std::endl;
        std::vector<at::Tensor> catFlattenMPSLossList;
        for (int k = 0; k < outAttrNum; k++) {
            catFlattenMPSLossList.push_back(
                at::cross_entropy_loss(
                    at::concat(flattenMPSPredLogitList[k], 0), // input
                    catFlattenMPSTarget.index({Ellipsis, k}), // target
                    {}, // weight
                    at::Reduction::None, // reduction
                    0LL // ignore_index
                )
            );
        }
        // calculate the sum of k attr losses
        at::Tensor catFlattenMPSLossStack = at::stack(catFlattenMPSLossList); // (outAttrNum, flattenMPSLength)
        // std::cout << catFlattenMPSLossStack << std::endl;
        // When an attribute is labeled as padding, its loss would be zero
        // but because we calculate loss by taking sum of the non-paddings of eachs heads
        // we can just take sum of them
        at::Tensor catFlattenMPSLossSum = catFlattenMPSLossStack.sum(0);
        // Find argmins and indices to replace
        std::vector<std::pair<int, int>> replaceIndices;
        int curFlattenMPSIndex = 0;
        for (size_t mpsNum = 0; mpsNum < curMPSIndices.size() - 1; mpsNum++) {
            int beginIndex = curMPSIndices[mpsNum];
            // if (beginIndex < 0) {
            //     continue;
            // }
            int endIndex = curMPSIndices[mpsNum+1];
            int mpsSize = endIndex - beginIndex;
            if (mpsSize == 2) {
                if (catFlattenMPSLossSum[curFlattenMPSIndex].item<float>()
                    > catFlattenMPSLossSum[curFlattenMPSIndex+1].item<float>()) {
                    replaceIndices.push_back(std::make_pair(beginIndex, beginIndex+1));
                    // std::cout << "(" << beginIndex << ", " << beginIndex+1 << "), ";
                }
                curFlattenMPSIndex += 2;
            }
            else if (mpsSize > 2) {
                int curFlattenMPSLength = mpsSize * (mpsSize + 1) / 2 - 1;
                // curMPSTriu = triu_indices(mpsSize, mpsSize, 0)[:, :-1] (drop last)
                at::Tensor curMPSTriu = at::triu_indices(mpsSize, mpsSize, 0).index({
                    Ellipsis, Slice(None, curFlattenMPSLength)
                });
                at::Tensor curStackedMPSLoss = at::full(
                    {mpsSize-1, mpsSize},
                    std::numeric_limits<float>::max(),
                    torch::TensorOptions().device(catFlattenMPSLossSum.device())
                );
                // make the flattened become stacked
                curStackedMPSLoss.index_put_(
                    {curMPSTriu.index({0}), curMPSTriu.index({1})},
                    catFlattenMPSLossSum.index({Slice(curFlattenMPSIndex, curFlattenMPSIndex+curFlattenMPSLength)})
                );
                curFlattenMPSIndex += curFlattenMPSLength;
                // std::cout << curStackedMPSLoss << std::endl;
                at::Tensor curMPSLossArgmin = curStackedMPSLoss.argmin(1);
                for (int i = 0; i < curMPSLossArgmin.size(0); i++) {
                    int minLossIndex = curMPSLossArgmin[i].item<int>();
                    if (minLossIndex != i) {
                        replaceIndices.push_back(std::make_pair(beginIndex+i, beginIndex+minLossIndex));
                        // std::cout << "(" << beginIndex+i << ", " << beginIndex+minLossIndex << "), ";
                    }
                }
            }
        }
        // std::cout << std::endl;
        
        // std::cout << "replaceIndices.size() " << replaceIndices.size() << std::endl;
        // modify the target such that the label at cur_index is replaced with the label at minLossIndex
        for (auto replacePair: replaceIndices) {
            // batchedTarget[batchNum, replacePair[0]] = batchedTarget[batchNum, replacePair[1]]
            batchedTarget.index_put_(
                {batchNum, replacePair.first},
                batchedTarget.index({batchNum, replacePair.second})
            );
        }
    }
    return batchedTarget;
}



at::Tensor findMinPermuLossTarget_Loop(
    std::vector<at::Tensor> batchedPredLogitsList,
    at::Tensor batchedTarget,
    std::vector<std::vector<int32_t>> batchedMPSIndices
) {
    torch::TensorOptions thisDevice =  torch::TensorOptions().device(batchedTarget.device());
    int64_t outAttrNum = batchedTarget.size(2);
    for (int64_t batchNum = 0; batchNum < batchedTarget.size(0); batchNum++) {
        // std::cout << "batchNum " << batchNum << std::endl;
        std::vector<int32_t> curMPSIndices = batchedMPSIndices[batchNum]; // copy
        
        std::vector<size_t> mpsNums;
        std::vector<int32_t> mpsSizes; 
        std::vector<int32_t> flattenMPSBeginIndices; 

        // Flatten MPSs
        // Calcucate the size of the flatten mps
        int32_t flattenMPSSize = 0;
        for (size_t mpsNum = 0; mpsNum < curMPSIndices.size() - 1; mpsNum++) {
            int32_t beginIndex = curMPSIndices[mpsNum];
            if (beginIndex < 0) {
                continue;
            }
            int32_t endIndex = curMPSIndices[mpsNum+1];
            int32_t mpsSize = endIndex - beginIndex;
            if (mpsSize < 0) {
                throw std::range_error("Negtive mpsSize");
            }
            else if (mpsSize == 2) {
                mpsNums.push_back(mpsNum);
                flattenMPSBeginIndices.push_back(flattenMPSSize);
                mpsSizes.push_back(mpsSize);
                flattenMPSSize += 2;
            }
            else if (mpsSize > 2) {
                mpsNums.push_back(mpsNum);
                flattenMPSBeginIndices.push_back(flattenMPSSize);
                mpsSizes.push_back(mpsSize);
                flattenMPSSize += mpsSize * (mpsSize + 1) / 2 - 1;
            }
        }
        // std::cout << "flattenMPSSize " << flattenMPSSize << std::endl;
        // for (auto mpsSize: mpsSizes) {
        //     std::cout << " " << mpsSize;
        // }
        // std::cout << std::endl;
        if (flattenMPSSize == 0) {
            continue;
        }

        // init flatten target and logits
        at::Tensor flattenMPSTarget = at::empty({flattenMPSSize, outAttrNum}, thisDevice.dtype(c10::ScalarType::Long));
        std::vector<at::Tensor> flattenMPSPredLogitList;
        for (int k = 0; k < outAttrNum; k++) {
            flattenMPSPredLogitList.push_back(at::empty({flattenMPSSize, batchedPredLogitsList[k].size(2)}, thisDevice));
        }
        for (size_t i = 0; i < mpsNums.size(); i++) {
            size_t mpsNum = mpsNums[i];
            int32_t beginIndex = curMPSIndices[mpsNum], endIndex = curMPSIndices[mpsNum+1];
            int32_t mpsSize = mpsSizes[i];
            int32_t curFlattenMPSBeginIndex = flattenMPSBeginIndices[i];
            if (mpsSize == 2) {
                int32_t curFlattenMPSEndIndex = curFlattenMPSBeginIndex + 2;
                flattenMPSTarget.index_put_(
                    {Slice(curFlattenMPSBeginIndex, curFlattenMPSEndIndex)},
                    batchedTarget.index({batchNum, Slice(beginIndex, endIndex)})
                );
                for (int k = 0; k < outAttrNum; k++) {
                    flattenMPSPredLogitList[k].index_put_(
                        {Slice(curFlattenMPSBeginIndex, curFlattenMPSEndIndex)},
                        batchedPredLogitsList[k].index({batchNum, beginIndex})
                    );
                }
            }
            else if (mpsSize > 2) {
                // curMPSTarget = batchedTarget[batch_number, begin_index:end_index]
                at::Tensor curMPSTarget = batchedTarget.index({batchNum, Slice(beginIndex, endIndex)});
                // we want:
                // t1, t2, t3 ... tn
                // -->
                // t1, t2, t3 ... tn, t2, t3, t4 ... tn, ... , tn-1, tn
                int32_t b = curFlattenMPSBeginIndex;
                for (int32_t j = 0; j < mpsSize - 1; j++) {
                    flattenMPSTarget.index_put_(
                        {Slice(b, b + mpsSize - j)},
                        curMPSTarget.index({Slice(j, mpsSize)})
                    );
                    b += mpsSize - j;
                }

                for (int k = 0; k < outAttrNum; k++) {
                    at::Tensor curMPSLogitK = batchedPredLogitsList[k].index({batchNum, Slice(beginIndex, endIndex)});
                    // we want:
                    // l1, l2, l3 ... ln
                    // -->
                    // l1, l1, l1, ... , l1 (n times), l2, l2, l2, ... , l2 (n-1 times), ... , ln-1, ln-1 (2 times)
                    int32_t b = curFlattenMPSBeginIndex;
                    for (int32_t j = 0; j < mpsSize - 1; j++) {
                        flattenMPSPredLogitList[k].index_put_(
                            {Slice(b, b + mpsSize - j)},
                            curMPSLogitK.index({j})
                        );
                        b += mpsSize - j;
                    }
                }
            }
        }

        std::vector<at::Tensor> flattenMPSLossesList;
        for (int k = 0; k < outAttrNum; k++) {
            flattenMPSLossesList.push_back(
                at::cross_entropy_loss(
                    at::concat(flattenMPSPredLogitList[k], 0), // input
                    flattenMPSTarget.index({Ellipsis, k}), // target
                    {}, // weight
                    at::Reduction::None, // reduction
                    0LL // ignore_index
                )
            );
        }
        // calculate the mean of k attr losses
        at::Tensor flattenMPSLossesStack = at::stack(flattenMPSLossesList); // (outAttrNum, flattenMPSLength)
        // std::cout << flattenMPSLossesStack << std::endl;
        // because when an attribute is labeled as padding, its loss would be zero
        // we want to ignore the zero values
        at::Tensor flattenMPSLossMean =
            flattenMPSLossesStack.sum(0) / flattenMPSLossesStack.count_nonzero(c10::optional<int64_t>(0));

        // Find argmins and indices to replace
        std::vector<std::pair<int, int>> replaceIndices;
        for (size_t i = 0; i < mpsNums.size(); i++) {
            size_t mpsNum = mpsNums[i];
            int32_t beginIndex = curMPSIndices[mpsNum];
            int32_t mpsSize = mpsSizes[i];
            int32_t curFlattenMPSBeginIndex = flattenMPSBeginIndices[i];
            if (mpsSize == 2) {
                if (flattenMPSLossMean[curFlattenMPSBeginIndex].item<float>()
                    > flattenMPSLossMean[curFlattenMPSBeginIndex+1].item<float>()) {
                    replaceIndices.push_back(std::make_pair(beginIndex, beginIndex+1));
                    // std::cout << "(" << beginIndex << ", " << beginIndex+1 << "), ";
                }
            }
            else if (mpsSize > 2) {
                int32_t b = curFlattenMPSBeginIndex;
                for (int32_t j = 0; j < mpsSize - 1; j++) {
                    int32_t m = flattenMPSLossMean.index({Slice(b, b + mpsSize - j)}).argmin().item<int32_t>();
                    if (m != j) {
                        replaceIndices.push_back(std::make_pair(beginIndex+j, beginIndex+m));
                        // std::cout << "(" << beginIndex+j << ", " << beginIndex+m << "), ";
                    }
                    b += mpsSize - j;
                }
            }
        }
        // std::cout << std::endl;
        
        // std::cout << "replaceIndices.size() " << replaceIndices.size() << std::endl;
        // modify the target such that the label at cur_index is replaced with the label at minLossIndex
        for (auto replacePair: replaceIndices) {
            // batchedTarget[batchNum, replacePair[0]] = batchedTarget[batchNum, replacePair[1]]
            batchedTarget.index_put_(
                {batchNum, replacePair.first},
                batchedTarget.index({batchNum, replacePair.second})
            );
        }
    }
    return batchedTarget;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ extension to make finding the modified target labels that minmize the the loss faster";
    m.def(
        "find_min_loss_target",
        &findMinPermuLossTarget_TriU,
        "For each position of predicted label, find the target label that minimizes the cross entropy loss in its MPS"
    );
}
