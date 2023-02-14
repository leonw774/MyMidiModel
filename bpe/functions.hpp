#ifndef FUNCS_H
#define FUNCS_H

#define COUNTING_THREAD_NUM 8

// return sum of all note's neighbor number
size_t updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict, unsigned int gapLimit);

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Shape>& shapeDict
);

double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreSingleton=false);

double calculateShapeEntropy(const Corpus& corpus);

double calculateAllAttributeEntropy(const Corpus& corpus);

template<typename T>
std::vector<std::pair<Shape, T>> shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& scoreFunc,
    const std::string& mergeCoundition,
    double samplingRate
);

template<typename T>
std::pair<Shape, T> findMaxValPair(const std::vector<std::pair<Shape, T>>& shapeScore);

#endif
