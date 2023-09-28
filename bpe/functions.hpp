#ifndef FUNCS_H
#define FUNCS_H

// return sum of all note's neighbor number
size_t updateNeighbor(Corpus& corpus, const std::vector<Shape>& shapeDict, unsigned int gapLimit);

Shape getShapeOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Shape>& shapeDict
);

double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreVelocity=false);

typedef std::vector<std::pair<Shape, unsigned int>> flatten_shape_counter_t;

flatten_shape_counter_t getShapeScore(
    Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    const std::string& adjacency,
    const double samplingRate
);

std::pair<Shape, unsigned int> findMaxValPair(const flatten_shape_counter_t& shapeScore);

#endif
