#ifndef SHAPE_COUNTING_H
#define SHAPE_COUNTING_H

#define COUNTING_THREAD_NUM 8

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const Shape& lShape,
    const Shape& rShape
);

void basicShapeCounting(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, int>& shapeCount,
    double samplingRate,
    double alpha
);

void pseudoRelevanceFeedbackInspiredShapeCounting(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::map<Shape, int>& shapeCount,
    int k,
    double samplingRate,
    double alpha
);

#endif
