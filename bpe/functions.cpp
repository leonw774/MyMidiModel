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
        if (g == 1) break;
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
int mergeCounters(shape_counter_t counterArray[], size_t arraySize) {
    std::vector<int> mergingMapIndices;
    for (int i = 0; i < arraySize; ++i) {
        mergingMapIndices.push_back(i);
    }
    while (mergingMapIndices.size() > 1) {
        // count from back to not disturb the index number when erasing
        #pragma omp parallel for
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


size_t updateNeighbor(
    Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    unsigned int gapLimit
) {
    size_t totalNeighborNumber = 0;
    // calculate the relative offset of all shapes in shapeDict
    std::vector<unsigned int> relOffsets(shapeDict.size(), 0);
    for (int t = 0; t < shapeDict.size(); ++t) {
        relOffsets[t] = getMaxRelOffset(shapeDict[t]);
    }
    // for each piece
    #pragma omp parallel for reduction(+: totalNeighborNumber)
    for (int i = 0; i < corpus.mns.size(); ++i) {
        // for each track
        for (Track& track: corpus.mns[i]) {
            // for each multinote
            for (int k = 0; k < track.size(); ++k) {
                // printTrack(corpus.piecesMN[i][j], shapeDict, k, 1);
                unsigned int onsetTime = track[k].onset;
                unsigned int offsetTime = onsetTime + (
                    relOffsets[track[k].shapeIndex] * track[k].stretch
                );
                unsigned int immdFollowOnset = -1;
                int n = 1;
                while (k+n < track.size() && n < MultiNote::neighborLimit) {
                    unsigned int nOnsetTime = track[k+n].onset;
                    // immediately following
                    if (nOnsetTime >= offsetTime) { 
                        if (immdFollowOnset == -1) {
                            // we want offsetTime <= nOnsetTime <= offsetTime + gap limit
                            if (nOnsetTime - offsetTime > gapLimit) {
                                break;
                            }
                            immdFollowOnset = nOnsetTime;
                        }
                        else if (nOnsetTime != immdFollowOnset) {
                            break;
                        }
                    }
                    // overlapping
                    // else {
                    //     /* do nothing */
                    // }
                    n++;
                }
                track[k].neighbor = n - 1;
                totalNeighborNumber += n - 1;
            }
        }
    }
    return totalNeighborNumber;
}

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Shape>& shapeDict
) {
    if (rmn.onset < lmn.onset) {
        // return getShapeOfMultiNotePair(rmn, lmn, shapeDict);
        throw std::runtime_error("right multi-note has smaller onset than left multi-note");
    }
    const Shape lShape = shapeDict[lmn.shapeIndex],
                rShape = shapeDict[rmn.shapeIndex];
    const int rightSize = rShape.size();
    const unsigned int rightStretch = rmn.stretch;

    // times = { right_onset_1, ..., right_onset_n, right_dur_1, ..., right_dur_n, left_stretch}
    unsigned int times[rightSize*2+1]; 
    int t = 0;
    for (const RelNote& rRelNote: rShape) {
        times[t] = (rRelNote.relOnset * rightStretch + rmn.onset - lmn.onset);
        times[t+rightSize] = (unsigned int) rRelNote.relDur * rightStretch;
        t++;
    }
    times[rightSize*2] = lmn.stretch;
    unsigned int newStretch = gcd(times, rightSize*2+1);
    unsigned int lStretchRatio = lmn.stretch / newStretch;

    // check to prevent overflow, because RelNote's onset is stored in uint8 and onsetLimit is 0x7f
    // if overflowed, return empty shape
    for (int i = 0; i < rightSize; ++i) {
        if (times[i] / newStretch > RelNote::onsetLimit) {
            return Shape();
        }
    }
    for (const RelNote& lRelNote: lShape) {
        if (lRelNote.relOnset * lStretchRatio > RelNote::onsetLimit) {
            return Shape();
        }
    }

    Shape pairShape;
    pairShape.reserve(lShape.size() + rShape.size());
    for (const RelNote& lRelNote: lShape) {
        pairShape.push_back(RelNote(
            lRelNote.relOnset * lStretchRatio,
            lRelNote.relPitch,
            lRelNote.relDur * lStretchRatio,
            lRelNote.isCont
        ));
    }
    t = 0;
    for (const RelNote& rRelNote: rShape) {
        pairShape.push_back(RelNote(
            times[t] / newStretch,
            rRelNote.relPitch + rmn.pitch - lmn.pitch,
            times[t+rightSize] / newStretch,
            rRelNote.isCont
        ));
        t++;
    }
    std::sort(pairShape.begin(), pairShape.end());
    return pairShape;
}

// ignoreVelcocity is default false
double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreVelcocity) {
    unsigned int max_thread_num = omp_get_max_threads();
    std::vector<uint8_t> multipiSizes[max_thread_num];
    #pragma omp parallel for
    for (int i = 0; i < corpus.mns.size(); ++i) {
        int thread_num = omp_get_thread_num();
        // for each track
        for (const Track &track: corpus.mns[i]) {
            // key: 64 bits: upper 16 unused, 8 for velocity, 8 for time stretch, 32 for onset
            // value: occurence count
            std::map<uint64_t, uint8_t> curTrackMulpiSizes;
            for (int k = 0; k < track.size(); ++k) {
                uint64_t key = track[k].onset;
                key |= ((uint64_t) track[k].stretch) << 32;
                if (!ignoreVelcocity) {
                    key |= ((uint64_t) track[k].vel) << 40;
                }
                curTrackMulpiSizes[key] += 1;
            }
            for (auto it = curTrackMulpiSizes.cbegin(); it != curTrackMulpiSizes.cend(); ++it) {
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


flatten_shape_counter_t getShapeScore(
    Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& adjacency,
    const double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in shapeScoring not in range (0, 1]");
    }
    bool isOursMerge = (adjacency == "ours");

    std::vector<unsigned int> samplePieceIndices;
    for (int i = 0; i < corpus.mns.size(); ++i) {
        if (samplingRate != 1.0) {
            if (((double) rand()) / RAND_MAX > samplingRate) continue;
        }
        samplePieceIndices.push_back(i);
    }

    unsigned int max_thread_num = omp_get_max_threads();
    shape_counter_t shapeScoreParallel[max_thread_num];
    #pragma omp parallel for
    for (int h = 0; h < samplePieceIndices.size(); ++h) {
        int i = samplePieceIndices[h];
        int thread_num = omp_get_thread_num();
        // to reduce the times we do "find" operations in big set
        shape_counter_t tempScoreDiff;
        // for each track
        for (const Track &track: corpus.mns[i]) {
            // for each multinote
            for (int k = 0; k < track.size(); ++k) {
                // for each neighbor
                for (int n = 1; n <= track[k].neighbor; ++n) {
                    if (isOursMerge) {
                        if (track[k].vel != track[k+n].vel) continue;
                    }
                    else {
                        // mulpi
                        if (track[k].onset != track[k+n].onset) break;
                        if (track[k].vel != track[k+n].vel) continue;
                        if (track[k].stretch != track[k+n].stretch) continue;
                    }
                    Shape shape = getShapeOfMultiNotePair(track[k], track[k+n], shapeDict);
                    // empty shape mean overflow happened
                    if (shape.size() == 0) continue;
                    tempScoreDiff[shape] += 1;
                }
            }
        }
        for (auto it = tempScoreDiff.cbegin(); it != tempScoreDiff.cend(); it++) {
            shapeScoreParallel[thread_num][it->first] += it->second;
        }
    }

    int mergedIndex = mergeCounters(shapeScoreParallel, max_thread_num);
    flatten_shape_counter_t shapeScore;
    shapeScore.reserve(shapeScoreParallel[mergedIndex].size());
    shapeScore.assign(
        shapeScoreParallel[mergedIndex].cbegin(),
        shapeScoreParallel[mergedIndex].cend()
    );
    return shapeScore;
}


std::pair<Shape, unsigned int> findMaxValPair(const flatten_shape_counter_t& shapeScore) {
    #pragma omp declare reduction\
        (maxsecond: std::pair<Shape, unsigned int>:\
            omp_out = omp_in.second > omp_out.second ? omp_in : omp_out\
        )
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

