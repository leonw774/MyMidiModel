#include "corpus.hpp"
#include "shapescore.hpp"
#include <string>
#include <algorithm>

int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 8 && argc != 9) {
        std::cout << "./learn_vocab inCorpusDirPath outCorpusDirPath bpeIter scoring mergeCondition samplingRate minScoreLimit (--verbose)" << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[1]);
    std::string outCorpusDirPath(argv[2]);
    int bpeIter = atoi(argv[3]);
    std::string scoring(argv[4]);
    std::string mergeCondition(argv[5]);
    double samplingRate = atof(argv[6]);
    double minScoreLimit = atof(argv[7]);
    bool verbose = false;
    if (argc == 9) {
        std::string v(argv[8]);
        if (v != "--verbose") {
            std::cout << "./learn_vocab inCorpusDirPath outCorpusDirPath bpeIter scoring mergeCondition samplingRate minScoreLimit (--verbose)" << std::endl;
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
        << "samplingRate: " << samplingRate << '\n'
        << "minScoreLimit: " << minScoreLimit << std::endl;

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
    std::cout << "Output merged corpus file: " << outCorpusFilePath << '\n'
        << "Reading input files" << std::endl;

    std::chrono::duration<double> onSencondDur = std::chrono::duration<double>(1);
    std::chrono::time_point<std::chrono::system_clock> programStartTimePoint = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> ioStartTimePoint = std::chrono::system_clock::now();

    // read parameters
    std::map<std::string, std::string> paras = readParasFile(parasFile);
    int nth, maxDur, maxTrackNum;
    std::string positionMethod;
    // stoi: c++11 thing
    nth = stoi(paras[std::string("nth")]);
    maxDur = stoi(paras[std::string("max_duration")]);
    maxTrackNum = stoi(paras[std::string("max_track_number")]);
    positionMethod = paras[std::string("position_method")];
    if (nth <= 0 || maxDur <= 0 || maxDur > 255 || maxTrackNum <= 0 || (positionMethod != "event" && positionMethod != "attribute")) {
        std::cout << "Corpus parameter error" << std::endl;
        return 1;
    }

    // read notes from corpus
    Corpus corpus = readCorpusFile(inCorpusFile, nth, positionMethod);
    std::cout << "Reading done. There are " << corpus.piecesTP.size() << " pieces" << std::endl;

    std::vector<Shape> shapeDict = getDefaultShapeDict();

    // sort and count notes
    size_t startMultinoteCount = 0, multinoteCount = 0;
    size_t drumMultinoteCount = 0;
    #pragma omp parallel for reduction(+: multinoteCount, drumMultinoteCount)
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
    double startAvgMulpi = calculateAvgMulpiSize(corpus, false);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start Multinote count: " << multinoteCount
        << ", Drum's multinote count: " << drumMultinoteCount
        << ", Start average mulpi: " << avgMulpi
        << ", Reading used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / onSencondDur << std::endl;

    if (multinoteCount == 0 || multinoteCount == drumMultinoteCount) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    std::vector<std::pair<Shape, unsigned int>> shapeScoreFreq;
    std::vector<std::pair<Shape, double>> shapeScoreWPlike;
    double neighborUpdatingTime, shapeScoringTime, findMaxTime, mergeTime;
    std::cout << "Iter, Found shapes count, Shape, Score, Multinote count, Iteration time, Neighbor updating time, Shape scoring time, Find max time, Merge time" << std::endl;
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        if (!verbose && iterCount != 0) 
            std::cout << "\33[2K\r"; // "\33[2K" is VT100 escape code that clear entire line
        std::cout << iterCount;
        std::chrono::time_point<std::chrono::system_clock>iterStartTimePoint = std::chrono::system_clock::now();
        std::chrono::time_point<std::chrono::system_clock>partStartTimePoint = std::chrono::system_clock::now();
        updateNeighbor(corpus, shapeDict);
        neighborUpdatingTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;

        // clac shape scores
        Shape maxScoreShape;
        if (scoring == "default") {
            partStartTimePoint = std::chrono::system_clock::now();
            shapeScoring<unsigned int>(corpus, shapeDict, shapeScoreFreq, scoring, mergeCondition, samplingRate, verbose);
            shapeScoringTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;
            partStartTimePoint = std::chrono::system_clock::now();
            std::pair<Shape, unsigned int> maxValPair = findMaxValPair(shapeScoreFreq);
            if (maxValPair.second < minScoreLimit) {
                std::cout << "\nEnd iterations because found best score < minScoreLimit\n";
                break;
            }
            maxScoreShape = maxValPair.first;
            std::cout << ", " << shapeScoreFreq.size() << ", " << "\"" << shape2str(maxScoreShape) << "\", " << maxValPair.second << ", ";
            shapeScoreFreq.clear();
            findMaxTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;
        }
        else {
            partStartTimePoint = std::chrono::system_clock::now();
            shapeScoring<double>(corpus, shapeDict, shapeScoreWPlike, scoring, mergeCondition, samplingRate, verbose);
            shapeScoringTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;
            partStartTimePoint = std::chrono::system_clock::now();
            std::pair<Shape, double> maxValPair = findMaxValPair(shapeScoreWPlike);
            if (maxValPair.second < minScoreLimit) {
                std::cout << "\nEnd iterations because found best score < minScoreLimit\n";
                break;
            }
            maxScoreShape = maxValPair.first;
            std::cout << ", " << shapeScoreFreq.size() << ", " << "\"" << shape2str(maxScoreShape) << "\", " << maxValPair.second << ", ";
            shapeScoreWPlike.clear();
            findMaxTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;
        }

        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxScoreShape);

        partStartTimePoint = std::chrono::system_clock::now();
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
                        if (corpus.piecesMN[i][j][k].vel == 0
                            || corpus.piecesMN[i][j][k+n].vel == 0
                            || corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) {
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
        #pragma omp parallel for reduction(+: multinoteCount)
        for (int i = 0; i < corpus.piecesMN.size(); ++i) {
            for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
                multinoteCount += corpus.piecesMN[i][j].size();
            }
        }
        mergeTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;

        std::cout << multinoteCount << ", ";
        std::cout << (std::chrono::system_clock::now() - iterStartTimePoint) / onSencondDur << ", "
            << neighborUpdatingTime << ", "
            << shapeScoringTime << ", "
            << findMaxTime << ", "
            << mergeTime;
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

    // Write files
    ioStartTimePoint = std::chrono::system_clock::now();
    writeShapeVocabFile(vocabFile, shapeDict);
    std::cout << "Writing merged corpus file" << std::endl;
    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum, positionMethod);
    std::cout << "Writing done. Writing used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / onSencondDur << '\n'
        << "Total used time: " << (std::chrono::system_clock::now() - programStartTimePoint) / onSencondDur << std::endl;
    return 0;
}
