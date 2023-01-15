#include <torch/extension.h>

#include <vector>
#include <utility>
#include <limits>
#include <iostream>

using namespace torch::indexing;

/*
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

    The modified target labels that minimizes the loss in contetxt of MPS
*/

at::Tensor findMinPermuLossTarget(
    std::vector<at::Tensor> batchedPredLogitsList,
    at::Tensor batchedTarget,
    std::vector<std::vector<int32_t>> batchedMPSIndices
) {
    torch::TensorOptions deviceOpt = torch::TensorOptions().device(batchedTarget.device());
    int outAttrNum = batchedPredLogitsList.size();
    for (int batchNum = 0; batchNum < batchedMPSIndices.size(); batchNum++) {
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
            if (beginIndex < 0) {
                continue;
            }
            int endIndex = curMPSIndices[mpsNum+1];
            int mpsSize = endIndex - beginIndex;
            if (mpsSize < 0) {
                throw std::range_error("Negtive mpsSize");
            }
            else if (mpsSize == 2) {
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
        // calculate the mean of k attr losses
        at::Tensor catFlattenMPSLossStack = at::stack(catFlattenMPSLossList); // (outAttrNum, flattenMPSLength)
        // std::cout << catFlattenMPSLossStack << std::endl;
        // because when an attribute is labeled as padding, its loss would be zero
        // we want to ignore the zero values
        at::Tensor catFlattenMPSLossMean =
            catFlattenMPSLossStack.sum(0) / catFlattenMPSLossStack.count_nonzero(c10::optional<int64_t>(0));
        // Find argmins and indices to replace
        std::vector<std::pair<int, int>> replaceIndices;
        int curFlattenMPSIndex = 0;
        for (size_t mpsNum = 0; mpsNum < curMPSIndices.size() - 1; mpsNum++) {
            int beginIndex = curMPSIndices[mpsNum];
            if (beginIndex < 0) {
                continue;
            }
            int endIndex = curMPSIndices[mpsNum+1];
            int mpsSize = endIndex - beginIndex;
            if (mpsSize == 2) {
                if (catFlattenMPSLossMean[curFlattenMPSIndex].item<float>()
                    > catFlattenMPSLossMean[curFlattenMPSIndex+1].item<float>()) {
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
                at::Tensor curMPSStackedLoss = at::full(
                    {mpsSize-1, mpsSize},
                    std::numeric_limits<float>::max(),
                    deviceOpt
                );
                // make the flattened become stacked
                curMPSStackedLoss.index_put_(
                    {curMPSTriu.index({0}), curMPSTriu.index({1})},
                    catFlattenMPSLossMean.index({Slice(curFlattenMPSIndex, curFlattenMPSIndex+curFlattenMPSLength)})
                );
                curFlattenMPSIndex += curFlattenMPSLength;
                // std::cout << curMPSStackedLoss << std::endl;
                at::Tensor curMPSLossArgmin = curMPSStackedLoss.argmin(1);
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

// torch::Tensor findMinPermuLossTarget(
//     std::vector<torch::Tensor> batchedPredLogitsList,
//     torch::Tensor batchedTarget,
//     std::vector<std::vector<int32_t>> batchedMPSIndices
// ) {
//     std::cout << "Number of batch: " << batchedMPSIndices.size();
//     std::cout << "\nbatchedMPSIndices: ";
//     for (size_t batchNum = 0; batchNum < batchedMPSIndices.size(); batchNum++) {
//         std::cout << batchedMPSIndices[batchNum].size() << ",";
//     }
//     std::cout << "\nbatchedPredLogitsList: ";
//     std::cout << "length:" << batchedPredLogitsList.size() << "\n";
//     for (size_t batchNum = 0; batchNum < batchedMPSIndices.size(); batchNum++) {
//         for (int i = 0; i < 3; i++) {
//             std::cout << batchedPredLogitsList[batchNum].size(i) << ",";
//         }
//         std::cout << "\n";
//     }
//     std::cout << "\nbatchedTarget: ";
//     for (int i = 0; i < 3; i++) {
//         std::cout << batchedTarget.size(i) << ",";
//     }
//     std::cout << std::endl;
//     return batchedTarget;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ extension to make finding the modified target labels that minmize the the loss faster";
    m.def(
        "find_min_loss_target",
        &findMinPermuLossTarget,
        "Find the target tensor that minimizes the loss in contetxt of MPS"
    );
}
