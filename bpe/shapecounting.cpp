#include "corpus.hpp"
#include "shapecounting.hpp"
#include <algorithm>
#include <random>
#include <queue>
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

/*
  Counting Methods

  - basicShapeCounting: count shapes in corpus, if samplingRate < 1, every track would have samplingRate of possibilty
    to be counted.

  - pseudoRelevanceFeedbackInspiredShapeCounting: count shape in randomly sampled corpus first,
    find the top-k of the result as candidate, and then count the candidate shapes in whole corpus

  alpha: every occurence of a shape increase the occurence count by (1 − alpha * (n−1)), instead of 1
  n is the index difference between the two merging multi-notes, and alpha is a adjustment parameter in [0, 1)
*/

void basicShapeCounting(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, int>& shapeCount,
    double samplingRate,
    double alpha
) {
    if (alpha < 0 || 1 <= alpha) {
        throw std::runtime_error("alpha in counting function not in range [0, 1)");
    }
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in counting function not in range (0, 1]");
    }
    std::map<Shape, int> shapeCountParallel[COUNTING_THREAD_NUM];
    std::map<Shape, double> shapeCountParallelDouble[COUNTING_THREAD_NUM];
    #pragma omp parallel for num_threads(COUNTING_THREAD_NUM)
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drums
            if (corpus.piecesTN[i][j] == 128) continue;
            // ignore by random
            if (samplingRate != 1.0) {
                if ((double) rand() / RAND_MAX > samplingRate) continue;
            }
            int thread_num = omp_get_thread_num();
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // for each neighbor
                for (int n = 1; n < corpus.piecesMN[i][j][k].neighbor; ++n) {
                    if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                    Shape s = getShapeOfMultiNotePair(
                        corpus.piecesMN[i][j][k],
                        corpus.piecesMN[i][j][k+n],
                        shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                        shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                    );
                    if (alpha == 0.0) {
                        if (shapeCountParallel[thread_num].count(s) == 0) {
                            shapeCountParallel[thread_num].insert(
                                std::pair<Shape, int>(s, (int) 1)
                            );
                        }
                        else {
                            shapeCountParallel[thread_num][s]++;
                        }
                    }
                    else {
                        double v = 1 - alpha * (n - 1);
                        if (shapeCountParallelDouble[thread_num].count(s) == 0) {
                            shapeCountParallelDouble[thread_num].insert(
                                std::pair<Shape, double>(s, v)
                            );
                        }
                        else {
                            shapeCountParallelDouble[thread_num][s] += v;
                        }
                    }
                }
            }
        }
    }
    // merge parrallel maps
    for (int j = 0; j < 8; ++j) {
        if (alpha == 0.0) {
            for (auto it = shapeCountParallel[j].begin(); it != shapeCountParallel[j].end(); it++) {
                if (shapeCount.count(it->first)) {
                    shapeCount[it->first] += it->second;
                }
                else {
                    shapeCount.insert(*it);
                }
            }
            shapeCountParallel[j].clear();
        }
        else {
            for (auto it = shapeCountParallelDouble[j].begin(); it != shapeCountParallelDouble[j].end(); it++) {
                if (shapeCount.count(it->first)) {
                    shapeCount[it->first] += it->second;
                }
                else {
                    shapeCount.insert(*it);
                }
            }
            shapeCountParallelDouble[j].clear();
        }
    }
}

void pseudoRelevanceFeedbackInspiredShapeCounting(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, int>& shapeCount,
    int k,
    double samplingRate,
    double alpha
) {
    if (samplingRate <= 0 || 1 <= samplingRate) {
        throw std::runtime_error("samplingRate in PRF-inspired counting not in range (0, 1)");
    }
    if (k <= 0) {
        throw std::runtime_error("k in PRF-inspired counting is less than zero.");
    }
    // first count
    std::map<Shape, int> firstShapeCount;
    if (k > 1) {
        basicShapeCounting(corpus, shapeDict, firstShapeCount, samplingRate, alpha);
    }
    else {
        // if k == 1, this is the same as basicShapeCounting
        basicShapeCounting(corpus, shapeDict, shapeCount, samplingRate, alpha);
        return;
    }
    // find top k
    std::priority_queue<std::pair<int, Shape>> pq;
    for (auto it = firstShapeCount.cbegin(); it != firstShapeCount.cend(); it++) {
        pq.push(std::pair<int, Shape>((*it).second, (*it).first));
    }
    std::set<Shape> topKShapes;
    for (int i = 0; i < k; ++i) {
        topKShapes.insert(pq.top().second);
        pq.pop();
    }
    // second count
    std::map<Shape, int> shapeCountParallel[COUNTING_THREAD_NUM];
    std::map<Shape, double> shapeCountParallelDouble[COUNTING_THREAD_NUM];
    #pragma omp parallel for num_threads(COUNTING_THREAD_NUM)
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drums
            if (corpus.piecesTN[i][j] == 128) continue;
            int thread_num = omp_get_thread_num();
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // for each neighbor
                for (int n = 1; n < corpus.piecesMN[i][j][k].neighbor; ++n) {
                    if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                    Shape s = getShapeOfMultiNotePair(
                        corpus.piecesMN[i][j][k],
                        corpus.piecesMN[i][j][k+n],
                        shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                        shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                    );
                    /********
                      Limit the counting to the top K shapes
                    ********/
                    if (topKShapes.count(s) == 0) continue;
                    if (alpha == 0.0) {
                        if (shapeCountParallel[thread_num].count(s) == 0) {
                            shapeCountParallel[thread_num].insert(
                                std::pair<Shape, int>(s, (int) 1)
                            );
                        }
                        else {
                            shapeCountParallel[thread_num][s]++;
                        }
                    }
                    else {
                        double v = 1 - alpha * (n - 1);
                        if (shapeCountParallelDouble[thread_num].count(s) == 0) {
                            shapeCountParallelDouble[thread_num].insert(
                                std::pair<Shape, double>(s, v)
                            );
                        }
                        else {
                            shapeCountParallelDouble[thread_num][s] += v;
                        }
                    }
                }
            }
        }
    }
    // merge parrallel maps
    for (int j = 0; j < 8; ++j) {
        if (alpha == 0.0) {
            for (auto it = shapeCountParallel[j].begin(); it != shapeCountParallel[j].end(); it++) {
                if (shapeCount.count(it->first)) {
                    shapeCount[it->first] += it->second;
                }
                else {
                    shapeCount.insert(*it);
                }
            }
            shapeCountParallel[j].clear();
        }
        else {
            for (auto it = shapeCountParallelDouble[j].begin(); it != shapeCountParallelDouble[j].end(); it++) {
                if (shapeCount.count(it->first)) {
                    shapeCount[it->first] += it->second;
                }
                else {
                    shapeCount.insert(*it);
                }
            }
            shapeCountParallelDouble[j].clear();
        }
    }
}