#include "corpus.hpp"
#include "shapecounting.hpp"
#include <string>
#include <algorithm>
#include <ctime>

void printTrack(const Track& track, const std::vector<Shape>& shapeDict, const size_t begin, const size_t length) {
    for (int i = begin; i < begin + length; ++i) {
        std::cout << i << " - Shape=" << shape2str(shapeDict[track[i].getShapeIndex()]);
        std::cout << " onset=" << (int) track[i].getOnset()
                << " basePitch=" << (int) track[i].pitch
                << " timeUnit=" << (int) track[i].unit
                << " velocity=" << (int) track[i].vel;
        std::cout << " neighbor=" << (int) track[i].neighbor << std::endl;
    }
}

int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 6) {
        std::cout << "Must have 5 arguments: [inCorpusDirPath] [outCorpusDirPath] [bpeIter] [countingMethod] [samplingRate]" << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[1]);
    std::string outCorpusDirPath(argv[2]);
    int bpeIter = atoi(argv[3]);
    std::string countingMethod(argv[4]);
    double samplingRate = atof(argv[5]);
    if (bpeIter <= 0 || 2045 < bpeIter) {
        std::cout << "Error: bpeIter <= 0 or > 2045: " << bpeIter << std::endl;
        return 1;
    }
    if (countingMethod != "ours" && countingMethod != "symphonynet" && countingMethod != "wordpiece") {
        std::cout << "Error: countingMethod is not ( ours | symphonynet | wordpiece ): " << countingMethod << std::endl;
        return 1;
    }
    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusDirPath: " << outCorpusDirPath << '\n'
        << "bpeIter: " << bpeIter << '\n'
        << "countingMethod: " << countingMethod << '\n'
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
    std::vector<Shape> shapeDict;
    shapeDict.reserve(bpeIter + 2);
    shapeDict.push_back({RelNote(0, 0, 0, 1)}); // DEFAULT_SHAPE_END
    shapeDict.push_back({RelNote(1, 0, 0, 1)}); // DEFAULT_SHAPE_CONT

    // sort and count notes
    size_t multinoteCount = 0;
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
    double avgMulpi = calculateAvgMulpiSize(corpus);
    // printTrack(corpus[0][0], shapeDict, 0, 10);

    std::cout << "Multinote count: " << multinoteCount
        << ", Drum's multinote count: " << drumMultinoteCount
        << ", Average mulpi: " << avgMulpi
        << ", Time: " << (unsigned int) time(0) - begTime << std::endl;

    if (multinoteCount == 0 || multinoteCount == drumMultinoteCount) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    time_t iterTime;
    std::map<Shape, unsigned int> shapeScore;
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        std::cout << "Iter:" << iterCount << ", ";
        iterTime = time(0);

        updateNeighbor(corpus, shapeDict);

        // count shape frequency
        if (countingMethod == "ours") {
            oursShapeCounting(corpus, shapeDict, shapeScore, samplingRate);
        }
        else if (countingMethod == "symphonynet") {
            symphonyNetShapeCounting(corpus, shapeDict, shapeScore, samplingRate);
        }
        else {
            wordPieceScoreShapeCounting(corpus, shapeDict, shapeScore, samplingRate);
        }
        std::cout << "Find " << shapeScore.size() << " unique pairs" << ", ";
        if (shapeScore.size() == 0) {
            std::cout << "Early termination: no shapes found" << std::endl;
            break;
        }
    
        // add shape with highest frequency into shapeDict
        Shape maxScoreShape;
        int maxScore = 0;
        for (auto it = shapeScore.cbegin(); it != shapeScore.cend(); it++) {
            if (maxScore < (*it).second) {
                maxScoreShape = (*it).first;
                maxScore = (*it).second;
            }
            // std::cout << shape2str((*it).first) << std::endl;
        }

        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxScoreShape);
        std::cout << "New shape: " << shape2str(maxScoreShape) << " Score=" << maxScore;

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
                // iterate forward
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
        std::cout << ". Multinote count: " << multinoteCount << ", ";
        
        shapeScore.clear();
        // corpus.shrink();
        std::cout << "Time: " << (unsigned int) time(0) - iterTime << std::endl;
    }

    avgMulpi = calculateAvgMulpiSize(corpus);
    std::cout << "Average mulpi: " << avgMulpi << std::endl;

    // write vocab file
    writeShapeVocabFile(vocabFile, shapeDict);
    
    std::cout << "Write vocabs file done.\n";
    std::cout << "Writing merged corpus file..." << std::endl;

    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum, positionMethod);

    std::cout << "Write merged corpus file done. Total used time: " << time(0)-begTime << std::endl;
    return 0;
}
