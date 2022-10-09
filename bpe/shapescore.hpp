#ifndef SHAPE_SCORING_H
#define SHAPE_SCORING_H

#define COUNTING_THREAD_NUM 8
#define IGNORE_DRUM true

void updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict, unsigned int gapLimit);

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Shape>& shapeDict
);

double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreSingleton=false);

template<typename T>
void shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::vector<std::pair<Shape, T>>& shapeScore,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate,
    bool verbose
);

template<typename T>
std::pair<Shape, T> findMaxValPair(const std::vector<std::pair<Shape, T>>& shapeScore);

#endif
