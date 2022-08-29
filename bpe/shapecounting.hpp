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

void oursShapeCounting(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, unsigned int>& shapeScore,
    double samplingRate
);

double calculateAvgMulpiSize(const Corpus& corpus);

void symphonyNetShapeCounting(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, unsigned int>& shapeScore,
    double samplingRate
);

void wordPieceScoreShapeCounting(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, unsigned int>& shapeScore,
    double samplingRate
);

#endif
