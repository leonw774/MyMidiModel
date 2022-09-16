#ifndef SHAPE_SCORING_H
#define SHAPE_SCORING_H

#define COUNTING_THREAD_NUM 8

void updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict);

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const Shape& lShape,
    const Shape& rShape
);

double calculateAvgMulpiSize(const Corpus& corpus);

template<typename T>
void shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::priority_queue<std::pair<T, Shape>>& shapeScore,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate
);

// #define SHAPE_SCORING_TEMP_IMPL
// #include "shapescore.tcc"

// use std::priority_queue<std::pair<unsigned int, Shape>> for shapeScore when scoringMethod is "default"
// use std::priority_queue<std::pair<double, Shape>> for shapeScore when scoringMethod is "wplike"
// template<>
// void shapeScoring<unsigned int>(
//     const Corpus& corpus,
//     const std::vector<Shape>& shapeDict,
//     std::priority_queue<std::pair<unsigned int, Shape>>& shapeScore,
//     const std::string& scoringMethod,
//     const std::string& mergeCoundition,
//     double samplingRate
// );

// template<>
// void shapeScoring<double>(
//     const Corpus& corpus,
//     const std::vector<Shape>& shapeDict,
//     std::priority_queue<std::pair<double, Shape>>& shapeScore,
//     const std::string& scoringMethod,
//     const std::string& mergeCoundition,
//     double samplingRate
// );


#endif
