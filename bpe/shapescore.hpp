#ifndef SHAPE_COUNTING_H
#define SHAPE_COUNTING_H

#define COUNTING_THREAD_NUM 8

void updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict);

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const Shape& lShape,
    const Shape& rShape
);

double calculateAvgMulpiSize(const Corpus& corpus);

void defaultShapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::priority_queue<std::pair<unsigned int, Shape>>& shapeScore,
    const std::string& mergeCoundition,
    double samplingRate
);

void wplikeShapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::priority_queue<std::pair<double, Shape>>& shapeScore,
    const std::string& mergeCoundition,
    double samplingRate
);

#endif
