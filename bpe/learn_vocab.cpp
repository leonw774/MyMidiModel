#include "corpus.hpp"
#include "shapescore.hpp"
#include <string>
#include <algorithm>
#include <unistd.h>
#include "omp.h"

int main(int argc, char *argv[]) {
    // read and validate args
    int cmd_opt = 0;
    bool clearLine = false;
    bool doLog = false;
    int nonOptStartIndex = 1;
    std::string cmdLineUsage = "./learn_vocab [-log] [-clearLine] inCorpusDirPath outCorpusDirPath bpeIter scoreFunc mergeCondition samplingRate minScoreLimit [workersNum]";
    while ((cmd_opt = getopt(argc, argv, "l:c:")) != -1) {
        nonOptStartIndex++;
        switch (cmd_opt) {
            case 'l':
                doLog = optarg;
                break;
            case 'c':
                clearLine = optarg;
                break;
            case '?':
                if (isprint(optopt)) {
                    std::cout << "Bad argument: " << argv[optopt] << "\n";
                }
                std::cout << cmdLineUsage << std::endl;
                return 1;
            default:
                std::cout << cmdLineUsage << std::endl;
                exit(1);
        }
    }
    if ((argc - nonOptStartIndex != 7) && (argc - nonOptStartIndex != 8)) {
        std::cout << "Bad number of non-optional arguments: " << argc - nonOptStartIndex << " != 7 or 8\n";
        for (int i = 0; i < argc; ++i) {
            std::cout << argv[i] << " ";
        }
        std::cout << "\n" << cmdLineUsage << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[nonOptStartIndex]);
    std::string outCorpusDirPath(argv[nonOptStartIndex+1]);
    int bpeIter = atoi(argv[nonOptStartIndex+2]);
    std::string scoreFunc(argv[nonOptStartIndex+3]);
    std::string mergeCondition(argv[nonOptStartIndex+4]);
    double samplingRate = atof(argv[nonOptStartIndex+5]);
    double minScoreLimit = atof(argv[nonOptStartIndex+6]);
    int workersNum = -1; // -1 means use default
    if (argc - nonOptStartIndex == 8) {
        workersNum = atoi(argv[nonOptStartIndex+7]);
        omp_set_num_threads(workersNum);
    }
    
    if (bpeIter <= 0 || MultiNote::shapeIndexLimit < bpeIter) {
        std::cout << "Error: bpeIter <= 0 or > 2045: " << bpeIter << std::endl;
        return 1;
    }
    if (scoreFunc != "freq" && scoreFunc != "wplike") {
        std::cout << "Error: scoreFunc is not \"freq\" or \"wplike\": " << scoreFunc << std::endl;
        return 1;
    }
    if (mergeCondition != "ours" && mergeCondition != "mulpi") {
        std::cout << "Error: mergeCondition is not \"ours\" or \"mulpi\": " << mergeCondition << std::endl;
        return 1;
    }
    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusDirPath: " << outCorpusDirPath << '\n'
        << "bpeIter: " << bpeIter << '\n'
        << "scoreFunc: " << scoreFunc << '\n'
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
    std::cout << "Output shape vocab file: " << vocabFilePath << std::endl;

    std::string outCorpusFilePath = outCorpusDirPath + "/corpus";
    std::cout << "Output merged corpus file: " << outCorpusFilePath << '\n'
        << "Reading input files" << std::endl;

    std::chrono::duration<double> oneSencondDur = std::chrono::duration<double>(1.0);
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
    corpus.sortAllTracks();

    size_t startMultinoteCount, multinoteCount;
    startMultinoteCount = multinoteCount = corpus.getMultiNoteCount();
    double startAvgMulpi = calculateAvgMulpiSize(corpus, false);
    double startShapeEntropy = calculateShapeEntropy(corpus);
    double startAllEntropy = calculateAllAttributeEntropy(corpus);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start Multinote count: " << multinoteCount
        << ", Start average mulpi: " << avgMulpi
        << ", Start shape entropy: " << startShapeEntropy
        << ", Start all attribute entropy: " << startAllEntropy
        << ", Reading used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / oneSencondDur << std::endl;

    if (multinoteCount == 0) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    std::vector<std::pair<Shape, unsigned int>> shapeScoreFreq;
    std::vector<std::pair<Shape, double>> shapeScoreWPlike;
    std::chrono::time_point<std::chrono::system_clock>iterStartTimePoint;
    std::chrono::time_point<std::chrono::system_clock>partStartTimePoint;
    double iterTime, findBestShapeTime, mergeTime, metricsTime = 0.0;
    double shapeEntropy = 0.0, allEntropy = 0.0;
    if (doLog) {
        std::cout << "Iter, Avg neighbor number, Found shapes count, Shape, Score, "
                  << "Multinote count, Shape entropy, All attribute entropy, "
                  << "Iteration time, Find best shape time, Merge time" << std::endl;
    }
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        iterStartTimePoint = std::chrono::system_clock::now();
        if (doLog) {
            if (clearLine && iterCount != 0) {
                std::cout << "\33[2K\r"; // "\33[2K" is VT100 escape code that clear entire line
            }
            std::cout << iterCount;
        }
        size_t totalNeighborNumber = updateNeighbor(corpus, shapeDict, nth); 

        // clac shape scores
        Shape maxScoreShape;
        partStartTimePoint = std::chrono::system_clock::now();
        if (scoreFunc == "freq") {
            shapeScoreFreq = shapeScoring<unsigned int>(corpus, shapeDict, scoreFunc, mergeCondition, samplingRate);
            std::pair<Shape, unsigned int> maxValPair = findMaxValPair(shapeScoreFreq);
            if (maxValPair.second < minScoreLimit) {
                std::cout << "\nEnd iterations because found best score < minScoreLimit\n";
                break;
            }
            maxScoreShape = maxValPair.first;
            if (doLog){
                std::cout << ", " << (double) totalNeighborNumber / multinoteCount << ", "
                        << shapeScoreFreq.size() << ", "
                        << "\"" << shape2str(maxScoreShape) << "\", "
                        << maxValPair.second << ", ";
            }
        }
        else {
            shapeScoreWPlike = shapeScoring<double>(corpus, shapeDict, scoreFunc, mergeCondition, samplingRate);
            std::pair<Shape, double> maxValPair = findMaxValPair(shapeScoreWPlike);
            if (maxValPair.second < minScoreLimit) {
                std::cout << "\nEnd iterations because found best score < minScoreLimit\n";
                break;
            }
            maxScoreShape = maxValPair.first;
            if (doLog){
                std::cout << ", " << (double) totalNeighborNumber / multinoteCount << ", "
                        << shapeScoreWPlike.size() << ", "
                        << "\"" << shape2str(maxScoreShape) << "\", "
                        << maxValPair.second << ", ";
            }
        }
        // check if shape has repeated relnote
        for (int i = 1; i < maxScoreShape.size(); ++i) {
            if (maxScoreShape[i] == maxScoreShape[i-1]) {
                std::cout << "Found invalid shape: " << shape2str(maxScoreShape) << "\n";
                return 1;
            } 
        }
        findBestShapeTime = (std::chrono::system_clock::now() - partStartTimePoint) / oneSencondDur;

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
                // for each multinote
                for (int k = 0; k < corpus.piecesMN[i][j].size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n <= corpus.piecesMN[i][j][k].neighbor; ++n) {
                        if (k + n >= corpus.piecesMN[i][j].size()) {
                            continue;
                        }
                        if (corpus.piecesMN[i][j][k].vel == 0
                            || corpus.piecesMN[i][j][k+n].vel == 0
                            || corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel) {
                            continue;
                        }
                        // if (shapeDict[corpus.piecesMN[i][j][k].shapeIndex].size()
                        //         + shapeDict[corpus.piecesMN[i][j][k+n].shapeIndex].size()
                        //         != maxScoreShape.size()) {
                        //     continue;
                        // }
                        Shape s = getShapeOfMultiNotePair(
                            corpus.piecesMN[i][j][k],
                            corpus.piecesMN[i][j][k+n],
                            shapeDict
                        );
                        if (s == maxScoreShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // the first relnote in the new shape is correspond to the first relnote in left multinote's original shape
                            uint8_t newUnit = shapeDict[corpus.piecesMN[i][j][k].shapeIndex][0].relDur * corpus.piecesMN[i][j][k].unit / maxScoreShape[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newUnit > maxDur) break;
                            corpus.piecesMN[i][j][k].unit = newUnit;
                            corpus.piecesMN[i][j][k].shapeIndex = newShapeIndex;

                            // mark right multinote to be removed by have vel set to 0
                            corpus.piecesMN[i][j][k+n].vel = 0;
                            break;
                        }
                    }
                }
                // "delete" multinotes with vel == 0
                // because back() could also to be removed, we iterate from back to front
                for (int k = corpus.piecesMN[i][j].size() - 1; k >= 0; --k) {
                    if (corpus.piecesMN[i][j][k].vel == 0) {
                        corpus.piecesMN[i][j][k] = corpus.piecesMN[i][j].back();
                        corpus.piecesMN[i][j].pop_back();
                    }
                }
                sort(corpus.piecesMN[i][j].begin(), corpus.piecesMN[i][j].end());
            }
        }
        mergeTime = (std::chrono::system_clock::now() - partStartTimePoint) / oneSencondDur;
        iterTime = (std::chrono::system_clock::now() - iterStartTimePoint) / oneSencondDur;
        if (doLog) {
            partStartTimePoint = std::chrono::system_clock::now();
            shapeEntropy = calculateShapeEntropy(corpus);
            allEntropy = calculateAllAttributeEntropy(corpus);
            multinoteCount = corpus.getMultiNoteCount();
            // To exclude the time used on calculating metrics
            metricsTime += (std::chrono::system_clock::now() - partStartTimePoint) / oneSencondDur;
            std::cout << multinoteCount << ", " << shapeEntropy << ", " << allEntropy << ", ";
            std::cout << iterTime << ", " << findBestShapeTime << ", " << mergeTime;
            if (clearLine)  std::cout.flush();
            else            std::cout << std::endl;
        }

        // corpus.shrink();
    }

    if (clearLine) {
        std::cout << '\n';
    }
    if (!doLog) {
        shapeEntropy = calculateShapeEntropy(corpus);
        allEntropy = calculateAllAttributeEntropy(corpus);
        multinoteCount = corpus.getMultiNoteCount();
    }
    avgMulpi = calculateAvgMulpiSize(corpus);
    std::cout << "End multinote count: " << multinoteCount
        << ", End average mulpi: " << avgMulpi
        << ", End shape entropy: " << shapeEntropy
        << ", End all attribute entropy: " << allEntropy
        << ", Multinote reduce rate: " << 1 - (double) multinoteCount / startMultinoteCount
        << ", Average mulpi reduce rate: " << 1 - avgMulpi / startAvgMulpi << std::endl;


    // Write files
    std::ofstream vocabFile(vocabFilePath, std::ios::out | std::ios::trunc);
    if (!vocabFile.is_open()) {
        std::cout << "Failed to open vocab output file: " << vocabFilePath << std::endl;
        return 1;
    }
    std::ofstream outCorpusFile(outCorpusFilePath, std::ios::out | std::ios::trunc);
    if (!outCorpusFile.is_open()) {
        std::cout << "Failed to open merged corpus output file: " << outCorpusFilePath << std::endl;
        return 1;
    }

    ioStartTimePoint = std::chrono::system_clock::now();
    writeShapeVocabFile(vocabFile, shapeDict);
    std::cout << "Writing merged corpus file" << std::endl;
    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum, positionMethod);
    std::cout << "Writing done. Writing used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / oneSencondDur << '\n'
        << "Total used time: " << (std::chrono::system_clock::now() - programStartTimePoint) / oneSencondDur - metricsTime << std::endl;
    return 0;
}
