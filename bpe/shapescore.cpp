#include "corpus.hpp"
#include "shapescore.hpp"
#include <algorithm>
#include <random>
#include "omp.h"

int gcd (int a, int b) {
    if (a == b) return a;
    if (a == 0 || b == 0) return ((a > b) ? a : b);
    if (a == 1 || b == 1) return 1;
    int tmp;
    while (b > 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

int gcd (int* arr, unsigned int size) {
    int g = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] != 0) {
            g = gcd(g, arr[i]);
        }
        if (g == 1) return 1;
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
                uint32_t onsetTime = corpus.piecesMN[i][j][k].getOnset();
                uint32_t maxRelOffset = findMaxRelOffset(shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()]);
                uint32_t offsetTime = corpus.piecesMN[i][j][k].getOnset() + maxRelOffset * corpus.piecesMN[i][j][k].unit;
                uint32_t immdAfterOnset = -1;
                int n = 1;
                while (k+n < corpus.piecesMN[i][j].size()) {
                    // overlapping
                    if ((corpus.piecesMN[i][j][k+n]).getOnset() < offsetTime) { 
                        n++;
                    }
                    // immediately after
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
    pairShape.resize(pairSize);

    int rightRelOnsets[rightSize];
    for (int i = 0; i < rightSize; ++i) {
        rightRelOnsets[i] = rShape[i].getRelOnset() * rmn.unit + rmn.getOnset() - lmn.getOnset();
    }
    int newUnit = gcd(gcd(lmn.unit, rmn.unit), gcd(rightRelOnsets, rightSize));
    // cout << "newUnit:" << newUnit << endl;
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
            pairShape[i].setRelOnset(rightRelOnsets[j] / newUnit);
            pairShape[i].relPitch = rShape[j].relPitch + rmn.pitch - lmn.pitch;
            pairShape[i].relDur = rShape[j].relDur * rmn.unit / newUnit;
            pairShape[i].setCont(rShape[j].isCont());
        }
    }
    std::sort(pairShape.begin(), pairShape.end());
    pairShape.shrink_to_fit();
    return pairShape;
}


double calculateAvgMulpiSize(const Corpus& corpus) {
    std::vector<uint8_t> multipiSizes;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drums
            if (corpus.piecesTP[i][j] == 128) continue;

            // key: 64 bits, upper 29 unused, 20 for onset, 8 for duration (time_unit), 7 for velocity
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

                uint64_t key = corpus.piecesMN[i][j][k].vel;
                key |= ((uint16_t) corpus.piecesMN[i][j][k].unit) << 7;
                key |= (corpus.piecesMN[i][j][k].getOnset() - corpus.piecesTS[i][measureCursor].onset) << 15;
                thisTrackMulpiSizes[key] += 1;
            }
            for (auto it = thisTrackMulpiSizes.cbegin(); it != thisTrackMulpiSizes.cend(); ++it) {
                multipiSizes.push_back(it->second);
            }
        }
    }
    size_t totalMulpiSize = 0;
    for (int i = 0; i < multipiSizes.size(); ++i) {
        totalMulpiSize += multipiSizes[i];
    }
    return (double) totalMulpiSize / (double) multipiSizes.size();
}


void defaultShapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::priority_queue<std::pair<unsigned int, Shape>>& shapeScore,
    const std::string& mergeCoundition,
    double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in oursShapeCounting not in range (0, 1]");
    }

    bool oursMerge = (mergeCoundition == "ours");
    std::map<Shape, unsigned int> shapeScoreParallel[COUNTING_THREAD_NUM];
    #pragma omp parallel for num_threads(COUNTING_THREAD_NUM)
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
                    if (oursMerge) {
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
                    shapeScoreParallel[thread_num][s] += 1;
                }
            }
        }
    }
    // merge parrallel maps
    for (int j = 1; j < 8; ++j) {
        for (auto it = shapeScoreParallel[j].cbegin(); it != shapeScoreParallel[j].cend(); it++) {
            shapeScoreParallel[0][it->first] += it->second;
        }
    }
    for (auto it = shapeScoreParallel[0].cbegin(); it != shapeScoreParallel[0].cend(); it++) {
        shapeScore.push(std::pair<unsigned int, Shape>(it->second, it->first));
    }
}

void wplikeShapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::priority_queue<std::pair<double, Shape>>& shapeScore,
    const std::string& mergeCoundition,
    double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in wordPieceScoreShapeCounting not in range (0, 1]");
    }

    std::vector<unsigned int> dictShapeCount(shapeDict.size(), 0);
    for (int i = 0; i < shapeDict.size(); ++i) {
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                    dictShapeCount[corpus.piecesMN[i][j][k].getShapeIndex()]++;
                }
            }
        }
    }
    std::map<Shape, double> shapeScoreParallel[COUNTING_THREAD_NUM];
    #pragma omp parallel for num_threads(COUNTING_THREAD_NUM)
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        int thread_num = omp_get_thread_num();
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drums
            if (corpus.piecesTP[i][j] == 128) continue;
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // for each neighbor
                for (int n = 1; n < corpus.piecesMN[i][j][k].neighbor; ++n) {
                    if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                    int lShapeIndex = corpus.piecesMN[i][j][k].getShapeIndex(),
                        rShapeIndex = corpus.piecesMN[i][j][k+n].getShapeIndex();
                    Shape s = getShapeOfMultiNotePair(
                        corpus.piecesMN[i][j][k],
                        corpus.piecesMN[i][j][k+n],
                        shapeDict[lShapeIndex],
                        shapeDict[rShapeIndex]
                    );
                    double v = 1 / (dictShapeCount[lShapeIndex] + dictShapeCount[rShapeIndex]);
                    shapeScoreParallel[thread_num][s] += v;
                }
            }
        }
    }
    // merge parrallel maps
    for (int j = 1; j < 8; ++j) {
        for (auto it = shapeScoreParallel[j].cbegin(); it != shapeScoreParallel[j].cend(); it++) {
            shapeScoreParallel[0][it->first] += it->second;
        }
    }
    for (auto it = shapeScoreParallel[0].cbegin(); it != shapeScoreParallel[0].cend(); it++) {
            shapeScore.push(std::pair<double, Shape>(it->second, it->first));
    }
}