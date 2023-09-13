#include "classes.hpp"
#include "functions.hpp"
#include <random>

#include <cmath>
#include <limits>
#include "omp.h"

unsigned int gcd(unsigned int a, unsigned int b) {
    if (a == b) return a;
    if (a == 0 || b == 0) return ((a > b) ? a : b);
    if (a == 1 || b == 1) return 1;
    // binary gcd
    // https://lemire.me/blog/2013/12/26/fastest-way-to-compute-the-greatest-common-divisor/
    // https://hbfs.wordpress.com/2013/12/10/the-speed-of-gcd/
    // use gcc build-in function __builtin_ctz
    unsigned int shift = __builtin_ctz(a|b);
    a >>= shift;
    do {
        b >>= __builtin_ctz(b);
        if (a > b) {
            std::swap(a, b);
        }
        b -= a;
    } while (b);
    return a << shift;
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
    Do merging in O(logt) time, t is arraySize
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


size_t updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict, unsigned int gapLimit) {
    size_t totalNeighborNumber = 0;
    // calculate the relative offset of all shapes in shapeDict
    std::vector<unsigned int> relOffsets(shapeDict.size(), 0);
    for (int t = 0; t < shapeDict.size(); ++t) {
        relOffsets[t] = getMaxRelOffset(shapeDict[t]);
    }
    // for each piece
    #pragma omp parallel for reduction(+: totalNeighborNumber)
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // printTrack(corpus.piecesMN[i][j], shapeDict, k, 1);
                unsigned int onsetTime = corpus.piecesMN[i][j][k].onset;
                unsigned int offsetTime = onsetTime + relOffsets[corpus.piecesMN[i][j][k].shapeIndex] * corpus.piecesMN[i][j][k].dur;
                unsigned int immdFollowOnset = -1;
                int n = 1;
                while (k+n < corpus.piecesMN[i][j].size() && n < MultiNote::neighborLimit) {
                    // overlapping
                    unsigned int nOnsetTime = (corpus.piecesMN[i][j][k+n]).onset;
                    if (nOnsetTime < offsetTime) { 
                        n++;
                    }
                    // immediately following
                    else if (immdFollowOnset == -1 || nOnsetTime == immdFollowOnset) {
                        n++;
                        // we want offsetTime <= nOnsetTime <= offsetTime + gap limit
                        if (nOnsetTime - offsetTime > gapLimit) {
                            break;
                        }
                        immdFollowOnset = nOnsetTime;
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
    if (rmn.onset < lmn.onset) {
        // return getShapeOfMultiNotePair(rmn, lmn, shapeDict);
        throw std::runtime_error("right multi-note has smaller onset than left multi-note");
    }
    Shape lShape = shapeDict[lmn.shapeIndex],
          rShape = shapeDict[rmn.shapeIndex];
    int leftSize = lShape.size(), rightSize = rShape.size();
    int pairSize = leftSize + rightSize;
    unsigned int leftDur = lmn.dur;
    unsigned int rightDur = rmn.dur;
    Shape pairShape;

    unsigned int unitAndOnsets[rightSize*2+1];
    for (int i = 0; i < rightSize; ++i) {
        unitAndOnsets[i]           = rShape[i].getRelOnset() * rightDur + rmn.onset - lmn.onset;
        unitAndOnsets[i+rightSize] = rShape[i].relDur * rightDur;
    }
    unitAndOnsets[rightSize*2] = lmn.dur;
    unsigned int newDur = gcd(unitAndOnsets, rightSize*2+1);

    // check to prevent overflow, because RelNote's onset is stored in uint8
    // if overflowed, return empty shape
    for (int i = 0; i < rightSize; ++i) {
        if (unitAndOnsets[i] / newDur > RelNote::onsetLimit) {
            return Shape();
        }
    }
    for (int i = 0; i < leftSize; ++i) {
        if (lShape[i].getRelOnset() * leftDur / newDur > RelNote::onsetLimit) {
            return Shape();
        }
    }
    pairShape.resize(pairSize);
    for (int i = 0; i < pairSize; ++i) {
        unsigned int temp = 0;
        if (i < leftSize) {
            if (i != 0) {
                pairShape[i].setRelOnset(lShape[i].getRelOnset() * leftDur / newDur);
                pairShape[i].relPitch = lShape[i].relPitch;
            }
            pairShape[i].relDur = lShape[i].relDur * leftDur / newDur;
            pairShape[i].setCont(lShape[i].isCont());
        }
        else {
            int j = i - leftSize;
            pairShape[i].setRelOnset(unitAndOnsets[j] / newDur);
            pairShape[i].relPitch = rShape[j].relPitch + rmn.pitch - lmn.pitch;
            pairShape[i].relDur = rShape[j].relDur * rightDur / newDur;
            pairShape[i].setCont(rShape[j].isCont());
        }
    }
    std::sort(pairShape.begin(), pairShape.end());
    return pairShape;
}


double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreVelcocity) { // ignoreVelcocity is default false
    unsigned int max_thread_num = omp_get_max_threads();
    std::vector<uint8_t> multipiSizes[max_thread_num];
    #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        int thread_num = omp_get_thread_num();
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // key: 64 bits: upper 16 unused, 8 for velocity, 8 for duration (time_unit), 32 for onset
            // value: occurence count
            std::map< uint64_t, uint8_t > thisTrackMulpiSizes;

            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                uint64_t key = corpus.piecesMN[i][j][k].onset;
                key |= ((uint64_t) corpus.piecesMN[i][j][k].dur) << 32;
                if (!ignoreVelcocity) {
                    key |= ((uint64_t) corpus.piecesMN[i][j][k].vel) << 40;
                }
                thisTrackMulpiSizes[key] += 1;
            }
            for (auto it = thisTrackMulpiSizes.cbegin(); it != thisTrackMulpiSizes.cend(); ++it) {
                multipiSizes[thread_num].push_back(it->second);
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


std::vector<std::pair<Shape, unsigned int>> shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& adjacency,
    double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in shapeScoring not in range (0, 1]");
    }
    bool isOursMerge = (adjacency == "ours");

    // std::chrono::time_point<std::chrono::system_clock> partStartTimePoint = std::chrono::system_clock::now();
    std::vector<std::pair<Shape, unsigned int>> shapeScore;

    std::vector<double> shapeFreq(shapeDict.size(), 0);
    size_t totalMultiNoteCount = 0;

    std::vector<unsigned int> samplePieceIndices;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        if (samplingRate != 1.0) {
            if (((double) rand()) / RAND_MAX > samplingRate) continue;
        }
        samplePieceIndices.push_back(i);
    }

    unsigned int max_thread_num = omp_get_max_threads();
    std::map<Shape, unsigned int> shapeScoreParallel[max_thread_num];
    #pragma omp parallel for
    for (int h = 0; h < samplePieceIndices.size(); ++h) {
        int i = samplePieceIndices[h];
        int thread_num = omp_get_thread_num();
        // to reduce the times we do "find" operations in big set
        std::map<Shape, unsigned int> tempScoreDiff;
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // for each neighbor
                for (int n = 1; n <= corpus.piecesMN[i][j][k].neighbor; ++n) {
                    if (isOursMerge) {
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                    }
                    else {
                        // mulpi
                        if (corpus.piecesMN[i][j][k].onset != corpus.piecesMN[i][j][k+n].onset) break;
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                        if (corpus.piecesMN[i][j][k].dur != corpus.piecesMN[i][j][k+n].dur) continue;
                    }
                    Shape s = getShapeOfMultiNotePair(
                        corpus.piecesMN[i][j][k],
                        corpus.piecesMN[i][j][k+n],
                        shapeDict
                    );
                    // empty shape mean overflow happened
                    if (s.size() == 0) continue;
                    // shapeScoreParallel[thread_num][s] += 1;
                    tempScoreDiff[s] += 1;
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


std::pair<Shape, unsigned int> findMaxValPair(const std::vector<std::pair<Shape, unsigned int>>& shapeScore) {
    #pragma omp declare reduction(maxsecond: std::pair<Shape, unsigned int> : omp_out = omp_in.second > omp_out.second ? omp_in : omp_out)
    std::pair<Shape, unsigned int> maxSecondPair;
    maxSecondPair.second = 0;
    #pragma omp parallel for reduction(maxsecond: maxSecondPair)
    for (int i = 0; i < shapeScore.size(); ++i) {
        if (shapeScore[i].second > maxSecondPair.second) {
            maxSecondPair = shapeScore[i];
        }
    }
    return maxSecondPair;
}

