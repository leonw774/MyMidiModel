#ifndef SHAPE_SCORING_H
#define SHAPE_SCORING_H

#define COUNTING_THREAD_NUM 8

// return sum of all note's neighbor number
size_t updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict, unsigned int gapLimit, bool ignoreDrum);

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Shape>& shapeDict
);

double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreDrum, bool ignoreSingleton=false);

double calculateLogLikelihood(const Corpus& corpus, bool ignoreDrum);

template<typename T>
void shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::vector<std::pair<Shape, T>>& shapeScore,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate,
    bool ignoreDrum,
    bool verbose
);

template<typename T>
std::pair<Shape, T> findMaxValPair(const std::vector<std::pair<Shape, T>>& shapeScore);

#endif
