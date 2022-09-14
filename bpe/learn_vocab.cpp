#include "corpus.hpp"
#include "shapescore.hpp"
#include <string>
#include <algorithm>
#include <ctime>

int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 7 && argc != 8) {
        std::cout << "./learn_vocab inCorpusDirPath outCorpusDirPath bpeIter scoring mergeCondition samplingRate (--verbose)" << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[1]);
    std::string outCorpusDirPath(argv[2]);
    int bpeIter = atoi(argv[3]);
    std::string scoring(argv[4]);
    std::string mergeCondition(argv[5]);
    double samplingRate = atof(argv[6]);
    bool verbose = false;
    if (argc == 8) {
        std::string v(argv[8]);
        if (v != "--verbose") {
            std::cout << "./learn_vocab inCorpusDirPath outCorpusDirPath bpeIter scoring mergeCondition samplingRate (--verbose)" << std::endl;
            return 1;
        }
        verbose = true;
    }
    if (bpeIter <= 0 || 2045 < bpeIter) {
        std::cout << "Error: bpeIter <= 0 or > 2045: " << bpeIter << std::endl;
        return 1;
    }
    if (scoring != "default" && scoring != "wplike") {
        std::cout << "Error: scoring is not ( default | wplike ): " << scoring << std::endl;
        return 1;
    }
    if (mergeCondition != "ours" && mergeCondition != "musicbpe") {
        std::cout << "Error: mergeCondition is not ( ours | musicbpe ): " << mergeCondition << std::endl;
        return 1;
    }
    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusDirPath: " << outCorpusDirPath << '\n'
        << "bpeIter: " << bpeIter << '\n'
        << "scoring: " << scoring << '\n'
        << "mergeCondition: " << mergeCondition << '\n'
        << "samplingRate: " << samplingRate << std::endl;

    // open files
    std::string inCorpusFilePath = inCorpusDirPath + "/corpus";
    std::ifstream inCorpusFile(inCorpusFilePath, std::ios::in | std::ios::binary);
    if (!inCorpusFile.is_open()) {
        std::cout << "Failed to open corpus file: " << inCorpusFilePath << std::endl;
        return 1;
    }
    std::cout << "Input corpus file: " << inCorpusFilePath << std::endl;
    
    std::string parasFilePath = inCorpusDirPath + "/paras";
    std::ifstream parasFile(parasFilePath, std::ios::in | std::ios::binary);
    if (!parasFile.is_open()) {
        std::cout << "Failed to open parameters file: " << parasFilePath << std::endl;
        return 1;
    }
    std::cout << "Input parameter file: " << parasFilePath << std::endl;

    std::string vocabFilePath = outCorpusDirPath + "/shape_vocab";
    std::ofstream vocabFile(vocabFilePath, std::ios::out | std::ios::trunc);
    if (!vocabFile.is_open()) {
        std::cout << "Failed to open vocab output file: " << vocabFilePath << std::endl;
        return 1;
    }
    std::cout << "Output shape vocab file: " << vocabFilePath << std::endl;

    std::string outCorpusFilePath = outCorpusDirPath + "/corpus";
    std::ofstream outCorpusFile(outCorpusFilePath, std::ios::out | std::ios::trunc);
    if (!outCorpusFile.is_open()) {
        std::cout << "Failed to open merged corpus output file: " << outCorpusFilePath << std::endl;
        return 1;
    }
    std::cout << "Output merged corpus file: " << outCorpusFilePath << std::endl;

    time_t begTime = time(0);
    std::cout << "Reading input files" << std::endl;

    // read parameters
    std::map<std::string, std::string> paras = readParasFile(parasFile);
    int nth, maxDur, maxTrackNum;
    std::string positionMethod;
    // stoi, c++11 thing
    nth = stoi(paras[std::string("nth")]);
    // std::cout << "nth: " << nth << std::endl;
    maxDur = stoi(paras[std::string("max_duration")]);
    // std::cout << "maxDur: " << maxDur << std::endl;
    maxTrackNum = stoi(paras[std::string("max_track_number")]);
    // std::cout << "maxTrackNum: " << maxTrackNum << std::endl;
    positionMethod = paras[std::string("position_method")];
    // std::cout << "positionMethod=" << positionMethod << std::endl;
    if (nth <= 0 || maxDur <= 0 || maxDur > 255 || maxTrackNum <= 0 || (positionMethod != "event" && positionMethod != "attribute")) {
        std::cout << "Corpus parameter error" << std::endl;
        return 1;
    }

    // read notes from corpus
    Corpus corpus = readCorpusFile(inCorpusFile, nth, positionMethod);
    std::vector<Shape> shapeDict = getDefaultShapeDict();

    // sort and count notes
    size_t startMultinoteCount = 0, multinoteCount = 0;
    size_t drumMultinoteCount = 0;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            if (corpus.piecesTP[i][j] == 128) {
                drumMultinoteCount += corpus.piecesMN[i][j].size();
            }
            multinoteCount += corpus.piecesMN[i][j].size();
            sort(corpus.piecesMN[i][j].begin(), corpus.piecesMN[i][j].end());
        }
    }
    startMultinoteCount = multinoteCount;
    double startAvgMulpi = calculateAvgMulpiSize(corpus);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start Multinote count: " << multinoteCount
        << ", Drum's multinote count: " << drumMultinoteCount
        << ", Start Average mulpi: " << avgMulpi
        << ", Read corpus time: " << (unsigned int) time(0) - begTime << std::endl;

    if (multinoteCount == 0 || multinoteCount == drumMultinoteCount) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    time_t iterTime;
    
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        if (verbose) std::cout << "Iter: " << iterCount << " ";
        else         std::cout << "\rIter: " << iterCount << '/' << bpeIter - 1 << " ";
        iterTime = time(0);

        updateNeighbor(corpus, shapeDict);

        // clac shape scores
        Shape maxScoreShape;
        if (scoring == "default") {
            std::priority_queue<std::pair<unsigned int, Shape>> shapeScore;
            defaultShapeScoring(corpus, shapeDict, shapeScore, mergeCondition, samplingRate);
            if (shapeScore.size() == 0) {
                std::cout << "Error: no shapes found" << std::endl;
                return 1;
            }
            std::cout << "Find " << shapeScore.size() << " unique pairs" << " ";
            int maxScore = shapeScore.top().first;
            maxScoreShape = shapeScore.top().second;
            std::cout << "New shape: " << shape2str(maxScoreShape) << " Score: " << maxScore << " ";
        }
        else {
            std::priority_queue<std::pair<double, Shape>> shapeScore;
            wplikeShapeScoring(corpus, shapeDict, shapeScore, mergeCondition, samplingRate);
            if (shapeScore.size() == 0) {
                std::cout << "Error: no shapes found" << std::endl;
                return 1;
            }
            std::cout << "Find " << shapeScore.size() << " unique pairs" << " ";
            double maxScore = shapeScore.top().first;
            maxScoreShape = shapeScore.top().second;
            std::cout << "New shape: " << shape2str(maxScoreShape) << " Score: " << maxScore << " ";
        }

        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxScoreShape);

        // merge MultiNotes with newly added shape
        // for each piece
        #pragma omp parallel for
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            // for each track
            #pragma omp parallel for
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                // ignore drum
                if (corpus.piecesTP[i][j] == 128) continue;
                // for each multinote
                for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n < corpus.piecesMN[i][j][k].neighbor; ++n) {
                        if (k + n >= corpus.piecesMN[i][j].size()) {
                            continue;
                        }
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel
                                || corpus.piecesMN[i][j][k].vel == 0
                                || corpus.piecesMN[i][j][k+n].vel == 0) {
                            continue;
                        }
                        // if (shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()].size()
                        //         + shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()].size()
                        //         != maxScoreShape.size()) {
                        //     continue;
                        // }
                        Shape s = getShapeOfMultiNotePair(
                            corpus.piecesMN[i][j][k],
                            corpus.piecesMN[i][j][k+n],
                            shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                            shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                        );
                        if (s == maxScoreShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // the first relnote in the new shape is correspond to the first relnote in left multinote's original shape
                            uint8_t newUnit = shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()][0].relDur * corpus.piecesMN[i][j][k].unit / maxScoreShape[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newUnit > maxDur) break;
                            corpus.piecesMN[i][j][k].unit = newUnit;
                            corpus.piecesMN[i][j][k].setShapeIndex(newShapeIndex);

                            // mark right multinote to be removed by have vel set to 0
                            corpus.piecesMN[i][j][k+n].vel = 0;
                            break;
                        }
                    }
                }
                // "delete" multinotes with vel == 0
                // to prevent that back() is also to be removed, we iterate from back to front
                for (int k = corpus.piecesMN[i][j].size() - 1; k >= 0; --k) {
                    if (corpus.piecesMN[i][j][k].vel == 0) {
                        corpus.piecesMN[i][j][k] = corpus.piecesMN[i][j].back();
                        corpus.piecesMN[i][j].pop_back();
                    }
                }
                sort(corpus.piecesMN[i][j].begin(), corpus.piecesMN[i][j].end());
            }
        }

        
        multinoteCount = 0;
        #pragma omp parallel for reduction(+:multinoteCount)
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                multinoteCount += corpus.piecesMN[i][j].size();
            }
        }
        std::cout << "Multinote count: " << multinoteCount << " ";
        std::cout << "Time: " << (unsigned int) time(0) - iterTime;
        if (verbose) std::cout << std::endl;
        else         std::cout.flush();

        // corpus.shrink();
    }
    if (!verbose) std::cout << '\n';

    multinoteCount = 0;
    #pragma omp parallel for reduction(+:multinoteCount)
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            multinoteCount += corpus.piecesMN[i][j].size();
        }
    }
    avgMulpi = calculateAvgMulpiSize(corpus);
    std::cout << "Ending multinote count: " << multinoteCount
        << ", Ending average mulpi: " << avgMulpi
        << ", Non-drum multinote reduce rate: " << 1 - (double) (multinoteCount - drumMultinoteCount) / (startMultinoteCount - drumMultinoteCount)
        << ", Average mulpi reduce rate: " << 1 - avgMulpi / startAvgMulpi << std::endl;

    // write vocab file
    writeShapeVocabFile(vocabFile, shapeDict);

    std::cout << "Writing merged corpus file" << std::endl;

    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum, positionMethod);

    std::cout << "Write merged corpus file done. Total used time: " << time(0)-begTime << std::endl;
    return 0;
}
