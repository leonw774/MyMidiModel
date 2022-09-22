#include "corpus.hpp"
#include "shapescore.hpp"
#include <algorithm>
#include <random>
#include "omp.h"

int gcd (unsigned int a, unsigned int b) {
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

int gcd (unsigned int* arr, unsigned int size) {
    int g = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] != 0) {
            g = gcd(g, arr[i]);
        }
    }
    return g;
}

void updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict) {
    // for each piece
    #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        #pragma omp parallel for
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drum
            if (corpus.piecesTP[i][j] == 128) continue;
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // printTrack(corpus.piecesMN[i][j], shapeDict, k, 1);
                unsigned int onsetTime = corpus.piecesMN[i][j][k].getOnset();
                unsigned int maxRelOffset = findMaxRelOffset(shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()]);
                unsigned int offsetTime = corpus.piecesMN[i][j][k].getOnset() + maxRelOffset * corpus.piecesMN[i][j][k].unit;
                unsigned int immdAfterOnset = -1;
                int n = 1;
                while (k+n < corpus.piecesMN[i][j].size()) {
                    // overlapping
                    if ((corpus.piecesMN[i][j][k+n]).getOnset() < offsetTime) { 
                        n++;
                    }
                    // immediately after
                    // else if ((immdAfterOnset == -1 || (corpus.piecesMN[i][j][k+n]).getOnset() == immdAfterOnset)
                    //     && (corpus.piecesMN[i][j][k+n]).getOnset() - offsetTime <= 4 * RelNote::onsetLimit) // add distance limit to immedAfter ) {
                    else if (immdAfterOnset == -1 || (corpus.piecesMN[i][j][k+n]).getOnset() == immdAfterOnset) {
                        immdAfterOnset = (corpus.piecesMN[i][j][k+n]).getOnset();
                        n++;
                    }
                    else {
                        break;
                    }
                }
                corpus.piecesMN[i][j][k].neighbor = n - 1;
            }
        }
    }
}

Shape getShapeOfMultiNotePair(const MultiNote& lmn, const MultiNote& rmn, const Shape& lShape, const Shape& rShape) {
    int leftSize = lShape.size(), rightSize = rShape.size();
    int pairSize = leftSize + rightSize;
    Shape pairShape;
    bool badShape = false;

    unsigned int unitAndOnsets[rightSize+2];
    unitAndOnsets[0] = lmn.unit;
    unitAndOnsets[1] = rmn.unit;
    for (int i = 2; i < rightSize+2; ++i) {
        unitAndOnsets[i] = rShape[i].getRelOnset() * rmn.unit + rmn.getOnset() - lmn.getOnset();
    }
    unsigned int newUnit = gcd(unitAndOnsets, rightSize+2);
    // checking to prevent overflow, because RelNote's onset has value limit
    for (int i = 2; i < rightSize+2; ++i) {
        if (unitAndOnsets[i] / newUnit > RelNote::onsetLimit) {
            badShape = true;
            break;
        }
    }
    if (!badShape) {
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
                int j = i - leftSize + 2;
                pairShape[i].setRelOnset(unitAndOnsets[j] / newUnit);
                pairShape[i].relPitch = rShape[j].relPitch + rmn.pitch - lmn.pitch;
                pairShape[i].relDur = rShape[j].relDur * rmn.unit / newUnit;
                pairShape[i].setCont(rShape[j].isCont());
            }
        }
        std::sort(pairShape.begin(), pairShape.end());
    }
    return pairShape;
}


double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreSingleton) {
    std::vector<uint8_t> multipiSizes;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drums
            if (corpus.piecesTP[i][j] == 128) continue;

            // key: 64 bits: upper 17 unused, 7 for velocity, 8 for duration (time_unit), 32 for onset
            // value: occurence count
            std::map< uint64_t, uint8_t > thisTrackMulpiSizes;

            // because in the paper it says 'position' (as in the measure), instead of onset (global time position)
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
                    multipiSizes.push_back(it->second);
                } 
            }
        }
    }
    size_t totalMulpiSize = 0;
    for (int i = 0; i < multipiSizes.size(); ++i) {
        totalMulpiSize += multipiSizes[i];
    }
    return (double) totalMulpiSize / (double) multipiSizes.size();
}

template<typename T>
void shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, T>& shapeScore,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in oursShapeCounting not in range (0, 1]");
    }
    bool isDefaultScoring = (scoringMethod == "default");
    bool isOursMerge = (mergeCoundition == "ours");

    std::vector<unsigned int> dictShapeCount(shapeDict.size(), 0);
    if (!isDefaultScoring) {
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                    dictShapeCount[corpus.piecesMN[i][j][k].getShapeIndex()]++;
                }
            }
        }
    }

    unsigned int max_thread_num = omp_get_max_threads();
    std::map<Shape, T> shapeScoreParallel[max_thread_num];
    #pragma omp parallel for
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        int thread_num = omp_get_thread_num();
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drums
            if (corpus.piecesTP[i][j] == 128) continue;
            // ignore by random
            if (samplingRate != 1.0) {
                if ((double) rand() / RAND_MAX > samplingRate) continue;
            }
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // for each neighbor
                for (int n = 1; n < corpus.piecesMN[i][j][k].neighbor; ++n) {
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
                        shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                        shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                    );
                    // empty shape is bad shape
                    if (s.size() == 0) continue;
                    if (isDefaultScoring) {
                        shapeScoreParallel[thread_num][s] += 1;
                    }
                    else {
                        unsigned int lShapeIndex = corpus.piecesMN[i][j][k].getShapeIndex(),
                                     rShapeIndex = corpus.piecesMN[i][j][k+n].getShapeIndex();
                        double v = 1 / (dictShapeCount[lShapeIndex] + dictShapeCount[rShapeIndex]);
                        shapeScoreParallel[thread_num][s] += v;
                    }
                }
            }
        }
    }
    // merge parrallel maps
    for (int j = 0; j < max_thread_num; ++j) {
        for (auto it = shapeScoreParallel[j].cbegin(); it != shapeScoreParallel[j].cend(); it++) {
            shapeScore[it->first] += it->second;
        }
    }
}

// instantiate function
// use std::priority_queue<std::pair<unsigned int, Shape>> for scoreShape when scoringMethod is "default"
// use std::priority_queue<std::pair<double, Shape>> for scoreShape when scoringMethod is "wplike"

template
void shapeScoring<unsigned int>(const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, unsigned int>& shapeScore,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate
);

template
void shapeScoring<double>(const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, double>& shapeScore,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate
);
