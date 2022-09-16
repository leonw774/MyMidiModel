#ifndef SHAPE_SCORING_TEMP_IMPL
#define SHAPE_SCORING_TEMP_IMPL

template<typename T> void shapeScoring(
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    std::priority_queue<std::pair<T, Shape>>& shapeScore,
    const std::string& scoringMethod,
    const std::string& mergeCoundition,
    double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error("samplingRate in oursShapeCounting not in range (0, 1]");
    }
    bool isDefaultScoring = (scoringMethod == "default");
    bool isOursMerge = (mergeCoundition == "ours");

    std::vector<unsigned int> dictShapeCount(shapeDict.size(), 0);
    if (isDefaultScoring) {
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                    dictShapeCount[corpus.piecesMN[i][j][k].getShapeIndex()]++;
                }
            }
        }
    }

    std::map<Shape, unsigned int> shapeScoreParallel[COUNTING_THREAD_NUM];
    #pragma omp parallel for num_threads(COUNTING_THREAD_NUM)
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        int thread_num = omp_get_thread_num();
        // for each track
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            // ignore drums
            if (corpus.piecesTP[i][j] == 128) continue;
            // ignore by random
            if (samplingRate != 1.0) {
                if ((double) rand() / RAND_MAX > samplingRate) continue;
            }
            // for each multinote
            for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                // for each neighbor
                for (int n = 1; n < corpus.piecesMN[i][j][k].neighbor; ++n) {
                    if (isOursMerge) {
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                    }
                    else {
                        if (corpus.piecesMN[i][j][k].getOnset() != corpus.piecesMN[i][j][k+n].getOnset()) break;
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) continue;
                        if (corpus.piecesMN[i][j][k].unit != corpus.piecesMN[i][j][k+n].unit) continue;
                    }
                    Shape s = getShapeOfMultiNotePair(
                        corpus.piecesMN[i][j][k],
                        corpus.piecesMN[i][j][k+n],
                        shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                        shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                    );
                    // empty shape is bad shape
                    if (s.size() > 0) continue;
                    if (isDefaultScoring) {
                        shapeScoreParallel[thread_num][s] += 1;
                    }
                    else {
                        unsigned int lShapeIndex = corpus.piecesMN[i][j][k].getShapeIndex(),
                                     rShapeIndex = corpus.piecesMN[i][j][k+n].getShapeIndex();
                        double v = 1 / (dictShapeCount[lShapeIndex] + dictShapeCount[rShapeIndex]);
                        shapeScoreParallel[thread_num][s] += v;
                    }
                }
            }
        }
    }
    // merge parrallel maps
    for (int j = 1; j < 8; ++j) {
        for (auto it = shapeScoreParallel[j].cbegin(); it != shapeScoreParallel[j].cend(); it++) {
            shapeScoreParallel[0][it->first] += it->second;
        }
    }
    for (auto it = shapeScoreParallel[0].cbegin(); it != shapeScoreParallel[0].cend(); it++) {
        shapeScore.push(std::pair<T, Shape>(it->second, it->first));
    }
}
