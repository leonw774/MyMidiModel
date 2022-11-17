#ifndef SHAPE_SCORING_H
#define SHAPE_SCORING_H

#define COUNTING_THREAD_NUM 8

// return sum of all note's neighbor number
size_t updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict, unsigned int gapLimit, bool excludeDrum);

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Shape>& shapeDict
);

double calculateAvgMulpiSize(const Corpus& corpus, bool excludeDrum, bool ignoreSingleton=false);

double calculateShapeEntropy(const Corpus& corpus, bool excludeDrum);

double calculateAllAttributeEntropy(const Corpus& corpus, bool excludeDrum);

template<typename T>
std::vector<std::pair<Shape, T>> shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& scoreFunc,
    const std::string& mergeCoundition,
    double samplingRate,
    bool excludeDrum
);

template<typename T>
std::pair<Shape, T> findMaxValPair(const std::vector<std::pair<Shape, T>>& shapeScore);

#endif
