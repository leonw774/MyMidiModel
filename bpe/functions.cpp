#include "classes.hpp"
#include "functions.hpp"
#include <random>

#include <cmath>
#include <limits>
#include "omp.h"

/*
    binary gcd
    https://hbfs.wordpress.com/2013/12/10/the-speed-of-gcd/
    use gcc build-in function __builtin_ctz
*/
unsigned int gcd(unsigned int a, unsigned int b) {
    if (a == b) return a;
    if (a == 0 || b == 0) return ((a > b) ? a : b);
    if (a == 1 || b == 1) return 1;
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
    for (int i = 1; i < size; ++i) {
        if (arr[i] != 0) {
            g = gcd(g, arr[i]);
            if (g == 1) break;
        }
    }
    return g;
}

/*  
    Do merging in O(logt) time, t is arraySize
    A merge between two counter with size of A and B takes
    $\sum_{i=A}^{A+B} \log{i}$ time
    Since $\int_A^{A+B} \log{x} dx = (A+B)\log{A+B} - A\log{A} - B$
    We could say a merge is O(n logn), where n is number of elements to count
    so that the total time complexity is O(n logn logt)

    This function alters the input array.
    The return index is the index of the counter merged all other counters.
*/
int mergeCounters(shape_counter_t counterArray[], size_t arraySize) {
    std::vector<int> mergingMapIndices(arraySize);
    for (int i = 0; i < arraySize; ++i) {
        mergingMapIndices[i] = i;
    }
    while (mergingMapIndices.size() > 1) {
        // count from back to not disturb the index number when erasing
        #pragma omp parallel for
        for (int i = mergingMapIndices.size() - 1; i > 0; i -= 2) {
            int a = mergingMapIndices[i];
            int b = mergingMapIndices[i-1];
            for (
                auto it = counterArray[a].cbegin();
                it != counterArray[a].cend();
                it++
            ) {
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
    for (int i = 0; i < corpus.pieceNum; ++i) {
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
        throw std::runtime_error(
            "right multi-note has smaller onset than left multi-note");
    }
    const Shape lShape = shapeDict[lmn.shapeIndex],
                rShape = shapeDict[rmn.shapeIndex];
    const int rSize = rShape.size(), lSize = lShape.size();
    const unsigned int rightStretch = rmn.stretch;
    const int32_t deltaOnset = (int32_t) rmn.onset - (int32_t) lmn.onset;

    // times = {
    //   right_onset_1, ..., right_onset_n,
    //   right_stretch_1, ..., right_stretch_n,
    //   left_stretch
    // }
    unsigned int times[rSize*2+1]; 
    int t = 0;
    for (const RelNote& rRelNote: rShape) {
        times[t] = rRelNote.relOnset * rightStretch + deltaOnset;
        times[t+rSize] = (unsigned int) rRelNote.relDur * rightStretch;
        t++;
    }
    times[rSize*2] = lmn.stretch;
    unsigned int newStretch = gcd(times, rSize*2+1);

    // re-calculate the new time values
    for (unsigned int& time: times) {
        time /= newStretch;
    }
    const unsigned int lStretchRatio = times[rSize*2];

    // check to prevent overflow, because RelNote's onset is stored in uint8
    // and onsetLimit is 0x7f
    // if overflowed, return empty shape
    for (int i = 0; i < rSize; ++i) {
        if (times[i] > RelNote::onsetLimit) {
            return Shape();
        }
    }
    unsigned int newLRelOnsets[lSize];
    unsigned int newLRelDur[lSize];
    t = 0;
    for (const RelNote& lRelNote: lShape) {
        newLRelOnsets[t] = lRelNote.relOnset * lStretchRatio;
        newLRelDur[t] = lRelNote.relDur * lStretchRatio;
        if (newLRelOnsets[t] > RelNote::onsetLimit
            || newLRelDur[t] > RelNote::durLimit) {
            return Shape();
        }
        t++;
    }

    const int8_t deltaPitch = (int8_t) rmn.pitch - (int8_t) lmn.pitch;
    // the merge part of merge sort
    int lpos = 0, rpos = 0;
    RelNote corLRelNote, curRRelNote;
    corLRelNote = RelNote(
        newLRelOnsets[0],
        lShape[0].relPitch,
        newLRelDur[0],
        lShape[0].isCont
    );
    curRRelNote = RelNote(
        times[0],
        rShape[0].relPitch + deltaPitch,
        times[rSize],
        rShape[0].isCont
    );
    Shape pairShape(lSize + rSize);
    enum MergeCase {LEFT, RIGHT};
    while (lpos < lSize || rpos < rSize) {
        int mergeCase = 0;
        if (rpos == rSize) {
            mergeCase = MergeCase::LEFT;
        }
        else if (lpos == lSize) {
            mergeCase = MergeCase::RIGHT;
        }
        else if (corLRelNote < curRRelNote) {
            mergeCase = MergeCase::LEFT;
        }
        else {
            mergeCase = MergeCase::RIGHT;
        }

        if (mergeCase == MergeCase::LEFT) {
            pairShape[lpos+rpos] = corLRelNote;
            lpos++;
            if (lpos != lSize) {
                corLRelNote = RelNote(
                    newLRelOnsets[lpos],
                    lShape[lpos].relPitch,
                    newLRelDur[lpos],
                    lShape[lpos].isCont
                );
            }
        }
        else {
            pairShape[lpos+rpos] = curRRelNote;
            rpos++;
            if (rpos != rSize) {
                curRRelNote = RelNote(
                    times[rpos],
                    rShape[rpos].relPitch + deltaPitch,
                    times[rpos+rSize],
                    rShape[rpos].isCont
                );
            }
        }
    }
    return pairShape;
}

// ignoreVelcocity is default false
double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreVelcocity) {
    unsigned int max_thread_num = omp_get_max_threads();
    std::vector<uint8_t> multipiSizes[max_thread_num];
    #pragma omp parallel for
    for (int i = 0; i < corpus.pieceNum; ++i) {
        int thread_num = omp_get_thread_num();
        std::vector<Track>& piece = corpus.mns[i];
        // for each track
        for (const Track &track: piece) {
            // key: 64 bits
            //     16 unused, 8 for velocity, 8 for time stretch, 32 for onset
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
            for (
                auto it = curTrackMulpiSizes.cbegin();
                it != curTrackMulpiSizes.cend();
                ++it
            ) {
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


flatten_shape_counter_t getShapeCounter(
    Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& adjacency,
    const double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error(
            "samplingRate in shapeScoring not in range (0, 1]");
    }
    bool isOursAdj = (adjacency == "ours");

    std::vector<unsigned int> samplePieceIndices;
    for (int i = 0; i < corpus.pieceNum; ++i) {
        if (samplingRate != 1.0) {
            if (((double) rand()) / RAND_MAX > samplingRate) continue;
        }
        samplePieceIndices.push_back(i);
    }

    unsigned int max_thread_num = omp_get_max_threads();
    shape_counter_t shapeCountersParallel[max_thread_num];
    #pragma omp parallel for
    for (int h = 0; h < samplePieceIndices.size(); ++h) {
        int i = samplePieceIndices[h];
        int thread_num = omp_get_thread_num();
        shape_counter_t& myShapeCounter = shapeCountersParallel[thread_num];
        std::vector<Track>& piece = corpus.mns[i];
        // for each track
        for (const Track& track: piece) {
            // for each multinote
            for (int k = 0; k < track.size(); ++k) {
                // for each neighbor
                const MultiNote& curMN = track[k];
                for (int n = 1; n <= curMN.neighbor; ++n) {
                    const MultiNote& neighborMN = track[k+n];
                    if (isOursAdj) {
                        if (curMN.vel != neighborMN.vel) continue;
                    }
                    else {
                        // mulpi
                        if (curMN.onset != neighborMN.onset) break;
                        if (curMN.vel != neighborMN.vel) continue;
                        if (curMN.stretch != neighborMN.stretch) continue;
                    }
                    Shape shape = getShapeOfMultiNotePair(
                        curMN,
                        neighborMN,
                        shapeDict
                    );
                    // empty shape mean overflow happened
                    if (shape.size() == 0) continue;
                    myShapeCounter[shape] += 1;
                }
            }
        }
    }

    int mergedIndex = mergeCounters(shapeCountersParallel, max_thread_num);
    flatten_shape_counter_t shapeCounter;
    shapeCounter.reserve(shapeCountersParallel[mergedIndex].size());
    shapeCounter.assign(
        shapeCountersParallel[mergedIndex].cbegin(),
        shapeCountersParallel[mergedIndex].cend()
    );
    return shapeCounter;
}


std::pair<Shape, unsigned int> findMaxValPair(
    const flatten_shape_counter_t& shapeCounter
) {
    #pragma omp declare reduction\
        (maxsecond: std::pair<Shape, unsigned int>:\
            omp_out = omp_in.second > omp_out.second ? omp_in : omp_out\
        )
    std::pair<Shape, unsigned int> maxSecondPair;
    maxSecondPair.second = 0;
    #pragma omp parallel for reduction(maxsecond: maxSecondPair)
    for (int i = 0; i < shapeCounter.size(); ++i) {
        if (shapeCounter[i].second > maxSecondPair.second) {
            maxSecondPair = shapeCounter[i];
        }
    }
    return maxSecondPair;
}

