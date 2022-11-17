#include "corpus.hpp"
#include "shapescore.hpp"
#include <random>
#include <cmath>
#include <limits>
#include "omp.h"

unsigned int gcd(unsigned int a, unsigned int b) {
    if (a == b) return a;
    if (a == 0 || b == 0) return ((a > b) ? a : b);
    if (a == 1 || b == 1) return 1;
    unsigned tmp;
    while (b > 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

unsigned int gcd(unsigned int* arr, unsigned int size) {
    int g = arr[0];
    for (int i = 1; i < size && g != 1; ++i) {
        if (arr[i] != 0) {
            g = gcd(g, arr[i]);
        }
    }
    return g;
}

/*  
    Do merging O(logt) in time, t is arraySize
    A merge between two counter with size of A and B takes $\sum_{i=A}^{A+B} \log{i}$ time
    Since $\int_A^{A+B} \log{x} dx = (A+B)\log{A+B} - A\log{A} - B$
    We could say a merge is O(n logn), where n is number of elements to count
    so that the total time complexity is O(n logn logt)

    This function alters the input array.
    The return index is the index of the counter merged all other counters. 
 */
template <typename T, typename S>
int mergeCounters(std::map<T, S> counterArray[], size_t arraySize) {
    std::vector<int> mergingMapIndices;
    for (int i = 0; i < arraySize; ++i) {
        mergingMapIndices.push_back(i);
    }
    while (mergingMapIndices.size() > 1) {
        #pragma omp parallel for
        // count from back to not disturb the index number when erasing
        for (int i = mergingMapIndices.size() - 1; i > 0; i -= 2) {
            int a = mergingMapIndices[i];
            int b = mergingMapIndices[i-1];
            for (auto it = counterArray[a].cbegin(); it != counterArray[a].cend(); it++) {
                counterArray[b][it->first] += it->second;
            }
        }
        for (int i = mergingMapIndices.size() - 1; i > 0; i -= 2) {
            mergingMapIndices.erase(mergingMapIndices.begin()+i);
        }
    }
    return mergingMapIndices[0];
}


size_t updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict, unsigned int gapLimit, bool excludeDrum) {
    size_t totalNeighborNumber = 0;
    // for each piece
    #pragma omp parallel for reduction(+: totalNeighborNumber)
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drum?
            if (corpus.piecesTP[i][j] == 128 && excludeDrum) continue;
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // printTrack(corpus.piecesMN[i][j], shapeDict, k, 1);
                unsigned int onsetTime = corpus.piecesMN[i][j][k].getOnset();
                unsigned int maxRelOffset = findMaxRelOffset(shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()]);
                unsigned int offsetTime = corpus.piecesMN[i][j][k].getOnset() + maxRelOffset * corpus.piecesMN[i][j][k].unit;
                unsigned int immdAfterOnset = -1;
                int n = 1;
                while (k+n < corpus.piecesMN[i][j].size() && n < MultiNote::neighborLimit) {
                    // overlapping
                    unsigned int nOnsetTime = (corpus.piecesMN[i][j][k+n]).getOnset();
                    if (nOnsetTime < offsetTime) { 
                        n++;
                    }
                    // immediately after
                    else if (immdAfterOnset == -1 || nOnsetTime == immdAfterOnset) {
                        n++;
                        // gap limit of immedAfter
                        if (nOnsetTime - offsetTime > gapLimit) {
                            break;
                        }
                        immdAfterOnset = nOnsetTime;
                    }
                    else {
                        break;
                    }
                }
                corpus.piecesMN[i][j][k].neighbor = n - 1;
                totalNeighborNumber += n - 1;
            }
        }
    }
    return totalNeighborNumber;
}

Shape getShapeOfMultiNotePair(const MultiNote& lmn, const MultiNote& rmn, const std::vector<Shape>& shapeDict) {
    Shape lShape = shapeDict[lmn.getShapeIndex()],
          rShape = shapeDict[rmn.getShapeIndex()];
    int leftSize = lShape.size(), rightSize = rShape.size();
    int pairSize = leftSize + rightSize;
    Shape pairShape;
    bool badShape = false;

    unsigned int unitAndOnsets[rightSize+2];
    for (int i = 0; i < rightSize; ++i) {
        unitAndOnsets[i] = rShape[i].getRelOnset() * rmn.unit + rmn.getOnset() - lmn.getOnset();
    }
    unitAndOnsets[rightSize] = lmn.unit;
    unitAndOnsets[rightSize+1] = rmn.unit;
    unsigned int newUnit = gcd(unitAndOnsets, rightSize+2);
    // check to prevent overflow, because RelNote's onset is stored in uint8
    for (int i = 0; i < rightSize; ++i) {
        if (unitAndOnsets[i] / newUnit > RelNote::onsetLimit) {
            badShape = true;
            break;
        }
    }
    if (badShape)
        // return empty shape
        return pairShape;
    pairShape.resize(pairSize);
    for (int i = 0; i < pairSize; ++i) {
        if (i < leftSize) {
            if (i != 0) {
                pairShape[i].setRelOnset(lShape[i].getRelOnset() * lmn.unit / newUnit);
                pairShape[i].relPitch = lShape[i].relPitch;
            }
            pairShape[i].relDur = lShape[i].relDur * lmn.unit / newUnit;
            pairShape[i].setCont(lShape[i].isCont());
        }
        else {
            int j = i - leftSize;
            pairShape[i].setRelOnset(unitAndOnsets[j] / newUnit);
            pairShape[i].relPitch = rShape[j].relPitch + rmn.pitch - lmn.pitch;
            pairShape[i].relDur = rShape[j].relDur * rmn.unit / newUnit;
            pairShape[i].setCont(rShape[j].isCont());
        }
    }
    std::sort(pairShape.begin(), pairShape.end());
    return pairShape;
}


double calculateAvgMulpiSize(const Corpus& corpus, bool excludeDrum, bool ignoreSingleton) {
    unsigned int max_thread_num = omp_get_max_threads();
    std::vector<uint8_t> multipiSizes[max_thread_num];
    #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        int thread_num = omp_get_thread_num();
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drum?
            if (corpus.piecesTP[i][j] == 128 && excludeDrum) continue;

            // key: 64 bits: upper 17 unused, 7 for velocity, 8 for duration (time_unit), 32 for onset
            // value: occurence count
            std::map< uint64_t, uint8_t > thisTrackMulpiSizes;

            unsigned int measureCursor = 0;
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // update measureCursor
                while (measureCursor < corpus.piecesTS[i].size() - 1) {
                    if (!corpus.piecesTS[i][measureCursor].getT()) {
                        if (corpus.piecesMN[i][j][k].getOnset() < corpus.piecesTS[i][measureCursor+1].onset) {
                            break;
                        }
                    }
                    measureCursor++;
                }

                uint64_t key = corpus.piecesMN[i][j][k].getOnset();
                key |= ((uint64_t) corpus.piecesMN[i][j][k].unit) << 32;
                key |= ((uint64_t) corpus.piecesMN[i][j][k].vel) << 40;
                thisTrackMulpiSizes[key] += 1;
            }
            for (auto it = thisTrackMulpiSizes.cbegin(); it != thisTrackMulpiSizes.cend(); ++it) {
                if (it->second > 1 or !ignoreSingleton) {
                    multipiSizes[thread_num].push_back(it->second);
                } 
            }
        }
    }
    size_t accumulatedMulpiSize = 0;
    size_t mulpiCount = 0;
    for (int i = 0; i < max_thread_num; ++i) {
        for (int j = 0; j < multipiSizes[i].size(); ++j) {
            accumulatedMulpiSize += multipiSizes[i][j];
        }
        mulpiCount += multipiSizes[i].size();
    }
    return (double) accumulatedMulpiSize / (double) mulpiCount;
}


std::vector<size_t> getShapeCounts(const Corpus& corpus, bool excludeDrum = false) {
    unsigned int max_thread_num = omp_get_max_threads();
    std::vector<size_t> shapeCountsParallel[max_thread_num];
    for (int i = 0; i < max_thread_num; ++i) {
        shapeCountsParallel[i].resize(MultiNote::shapeIndexLimit);
    }
    #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        int thread_num = omp_get_thread_num();
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drum?
            if (corpus.piecesTP[i][j] == 128 && excludeDrum) continue;
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                shapeCountsParallel[thread_num][corpus.piecesMN[i][j][k].getShapeIndex()] += 1;
            }
        }
    }
    // mergeCounters, but for vectors
    std::vector<int> mergingMapIndices;
    for (int i = 0; i < max_thread_num; ++i) {
        mergingMapIndices.push_back(i);
    }
    while (mergingMapIndices.size() > 1) {
        #pragma omp parallel for
        // count from back to not disturb the index number when erasing
        for (int i = mergingMapIndices.size() - 1; i > 0; i -= 2) {
            int a = mergingMapIndices[i];
            int b = mergingMapIndices[i-1];
            for (int i = 0; i < MultiNote::shapeIndexLimit; ++i) {
                shapeCountsParallel[b][i] += shapeCountsParallel[a][i];
            }
        }
        for (int i = mergingMapIndices.size() - 1; i > 0; i -= 2) {
            mergingMapIndices.erase(mergingMapIndices.begin()+i);
        }
    }
    return shapeCountsParallel[mergingMapIndices[0]];
}


double calculateShapeEntropy(const Corpus& corpus, bool excludeDrum) {
    std::vector<size_t> shapeCounts = getShapeCounts(corpus, excludeDrum);
    size_t totalCount = 0;
    for (int i = 0; i < shapeCounts.size(); ++i) {
        totalCount += shapeCounts[i];
    }
    double logTotlaCount = log2(totalCount);
    double entropy = 0;
    std::vector<double> shapeFreq(shapeCounts.size(), 0.0);
    for (int i = 0; i < shapeFreq.size(); ++i) {
        if (shapeCounts[i] != 0) {
            entropy -= shapeCounts[i] * ((log2(shapeCounts[i]) - logTotlaCount) / totalCount);
        }
    }
    return entropy;
}


double calculateAllAttributeEntropy(const Corpus& corpus, bool excludeDrum) {
    unsigned int max_thread_num = omp_get_max_threads();
    std::map<uint64_t, unsigned int> allAttrCountParallel[max_thread_num];
    size_t totalCount = 0;
    // array of shape index, pitch, unit, velocity
    #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        int thread_num = omp_get_thread_num();
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drum?
            if (corpus.piecesTP[i][j] == 128 && excludeDrum) continue;
            totalCount += corpus.piecesMN[i][j].size();
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                uint64_t key = corpus.piecesMN[i][j][k].getShapeIndex();
                key |= ((uint64_t) corpus.piecesMN[i][j][k].pitch) << 16;
                key |= ((uint64_t) corpus.piecesMN[i][j][k].unit)  << 24;
                key |= ((uint64_t) corpus.piecesMN[i][j][k].vel)   << 32;
                allAttrCountParallel[thread_num][key] += 1;
            }
        }
    }
    int mergedIndex = mergeCounters(allAttrCountParallel, max_thread_num);
    double logTotlaCount = log2(totalCount);
    double entropy = 0;
    for (auto it = allAttrCountParallel[mergedIndex].begin(); it != allAttrCountParallel[mergedIndex].end(); ++it) {
        unsigned int count = it->second;
        entropy -= count * ((log2(count) - logTotlaCount) / totalCount);
    }
    return entropy;
}


template<typename T>
std::vector<std::pair<Shape, T>> shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& scoreFunc,
    const std::string& mergeCoundition,
    double samplingRate,
    bool excludeDrum
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in shapeScoring not in range (0, 1]");
    }
    bool isFreqScore = (scoreFunc == "freq");
    bool isOursMerge = (mergeCoundition == "ours");

    // std::chrono::time_point<std::chrono::system_clock> partStartTimePoint = std::chrono::system_clock::now();
    std::vector<std::pair<Shape, T>> shapeScore;

    std::vector<double> shapeFreq(shapeDict.size(), 0);
    size_t totalMultiNoteCount = 0;
    if (!isFreqScore) {
        std::vector<size_t> shapeCounts = getShapeCounts(corpus, excludeDrum);
        if (shapeCounts.size() < shapeDict.size()) {
            shapeCounts.resize(shapeDict.size(), 0);
        }
        for (int i = 0; i < shapeCounts.size(); ++i) {
            totalMultiNoteCount += shapeCounts[i];
        }
        for (int i = 0; i < shapeFreq.size(); ++i) {
            shapeFreq[i] = ((double) shapeCounts[i]) / totalMultiNoteCount;
        }
    }

    std::vector<unsigned int> samplePieceIndices;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        if (samplingRate != 1.0) {
            if (((double) rand()) / RAND_MAX > samplingRate) continue;
        }
        samplePieceIndices.push_back(i);
    }

    unsigned int max_thread_num = omp_get_max_threads();
    std::map<Shape, T> shapeScoreParallel[max_thread_num];
    #pragma omp parallel for
    for (int h = 0; h < samplePieceIndices.size(); ++h) {
        int i = samplePieceIndices[h];
        int thread_num = omp_get_thread_num();
        // to reduce the times we do "find" operations in big set
        std::map<Shape, T> tempScoreDiff;
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drum?
            if (corpus.piecesTP[i][j] == 128 && excludeDrum) continue;
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // for each neighbor
                for (int n = 1; n <= corpus.piecesMN[i][j][k].neighbor; ++n) {
                    if (isOursMerge) {
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                    }
                    else {
                        if (corpus.piecesMN[i][j][k].getOnset() != corpus.piecesMN[i][j][k+n].getOnset()) break;
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                        if (corpus.piecesMN[i][j][k].unit != corpus.piecesMN[i][j][k+n].unit) continue;
                    }
                    Shape s = getShapeOfMultiNotePair(
                        corpus.piecesMN[i][j][k],
                        corpus.piecesMN[i][j][k+n],
                        shapeDict
                    );
                    // empty shape is bad shape
                    if (s.size() == 0) continue;
                    if (isFreqScore) {
                        // shapeScoreParallel[thread_num][s] += 1;
                        tempScoreDiff[s] += 1;
                    }
                    else {
                        unsigned int lShapeIndex = corpus.piecesMN[i][j][k].getShapeIndex(),
                                     rShapeIndex = corpus.piecesMN[i][j][k+n].getShapeIndex();
                        tempScoreDiff[s] += 1 / (shapeFreq[lShapeIndex] * shapeFreq[rShapeIndex]);
                        // shapeScoreParallel[thread_num][s] += 1.0 / (shapeFreq[lShapeIndex] + shapeFreq[rShapeIndex]);
                    }
                }
            }
        }
        for (auto it = tempScoreDiff.cbegin(); it != tempScoreDiff.cend(); it++) {
            shapeScoreParallel[thread_num][it->first] += it->second;
        }
    }
    // std::cout << " shape scoring time=" << (std::chrono::system_clock::now() - partStartTimePoint) / std::chrono::duration<double>(1) << ' ';
    // partStartTimePoint = std::chrono::system_clock::now();
    int mergedIndex = mergeCounters(shapeScoreParallel, max_thread_num);
    shapeScore.reserve(shapeScoreParallel[mergedIndex].size());
    shapeScore.assign(shapeScoreParallel[mergedIndex].cbegin(), shapeScoreParallel[mergedIndex].cend());
    // std::cout << "merge mp result time=" << (std::chrono::system_clock::now() - partStartTimePoint) / std::chrono::duration<double>(1);
    return shapeScore;
}

template<typename T>
std::pair<Shape, T> findMaxValPair(const std::vector<std::pair<Shape, T>>& shapeScore) {
    #pragma omp declare reduction(maxsecond : std::pair<Shape, T> : omp_out = omp_in.second > omp_out.second ? omp_in : omp_out)
    std::pair<Shape, T> maxSecondPair;
    maxSecondPair.second = 0;
    #pragma omp parallel for reduction(maxsecond : maxSecondPair)
    for (int i = 0; i < shapeScore.size(); ++i) {
        if (shapeScore[i].second > maxSecondPair.second) {
            maxSecondPair = shapeScore[i];
        }
    }
    return maxSecondPair;
}

// instantiate function
// use <unsigned int> when scoringMethod is "default"
// use <double> when scoringMethod is "wplike"

template
std::vector<std::pair<Shape, unsigned int>> shapeScoring<unsigned int>(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate,
    bool excludeDrum
);

template
std::vector<std::pair<Shape, double>> shapeScoring<double>(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate,
    bool excludeDrum
);

template
std::pair<Shape, unsigned int> findMaxValPair(const std::vector<std::pair<Shape, unsigned int>>& shapeScore);

template
std::pair<Shape, double> findMaxValPair(const std::vector<std::pair<Shape, double>>& shapeScore);
