#include "corpus.hpp"
#include "shapescore.hpp"
#include <string>
#include <algorithm>
#include <unistd.h>

int main(int argc, char *argv[]) {
    // read and validate args
    int cmd_opt = 0;
    bool verbose = false;
    bool ignoreDrum = false;
    while ((cmd_opt = getopt(argc, argv, "v:i:")) != -1) {
        switch (cmd_opt) {
            case 'v':
                verbose = optarg;
                break;
            case 'i':
                ignoreDrum = optarg;
                break;
            case '?':
                if (isprint(optopt)) {
                    std::cout << "Bad argument: " << argv[optopt] << "\n";
                }
                std::cout << "./learn_vocab [-verbose] [-ignoredrum] inCorpusDirPath outCorpusDirPath bpeIter scoring mergeCondition samplingRate minScoreLimit" << std::endl;
                return 1;
            default:
                std::cout << "./learn_vocab [-verbose] [-ignoredrum] inCorpusDirPath outCorpusDirPath bpeIter scoring mergeCondition samplingRate minScoreLimit" << std::endl;
                exit(1);
        }
    }
    int nonOptStartIndex = optind;
    std::cout << nonOptStartIndex << " " << argc << "\n";
    if (argc - nonOptStartIndex != 7) {
        std::cout << "./learn_vocab [-verbose] [-ignoredrum] inCorpusDirPath outCorpusDirPath bpeIter scoring mergeCondition samplingRate minScoreLimit" << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[nonOptStartIndex]);
    std::string outCorpusDirPath(argv[nonOptStartIndex+1]);
    int bpeIter = atoi(argv[nonOptStartIndex+2]);
    std::string scoring(argv[nonOptStartIndex+3]);
    std::string mergeCondition(argv[nonOptStartIndex+4]);
    double samplingRate = atof(argv[nonOptStartIndex+5]);
    double minScoreLimit = atof(argv[nonOptStartIndex+6]);
    
    if (bpeIter <= 0 || 2045 < bpeIter) {
        std::cout << "Error: bpeIter <= 0 or > 2045: " << bpeIter << std::endl;
        return 1;
    }
    if (scoring != "default" && scoring != "wplike") {
        std::cout << "Error: scoring is not ( default | wplike ): " << scoring << std::endl;
        return 1;
    }
    if (mergeCondition != "ours" && mergeCondition != "mulpi") {
        std::cout << "Error: mergeCondition is not ( ours | mulpi ): " << mergeCondition << std::endl;
        return 1;
    }
    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusDirPath: " << outCorpusDirPath << '\n'
        << "bpeIter: " << bpeIter << '\n'
        << "scoring: " << scoring << '\n'
        << "mergeCondition: " << mergeCondition << '\n'
        << "samplingRate: " << samplingRate << '\n'
        << "minScoreLimit: " << minScoreLimit << '\n'
        << "IGNORE_DRUM: " << (IGNORE_DRUM ? "true" : "false") << std::endl;

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
    corpus.sortAllTracks();
    size_t startMultinoteCount, multinoteCount, drumMultinoteCount;
    startMultinoteCount = multinoteCount = corpus.getMultiNoteCount();
    drumMultinoteCount = corpus.getMultiNoteCount(true);
    double startAvgMulpi = calculateAvgMulpiSize(corpus, ignoreDrum, false);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start Multinote count: " << multinoteCount
        << ", Drum's multinote count: " << drumMultinoteCount
        << ", Start average mulpi: " << avgMulpi
        << ", Reading used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / onSencondDur << std::endl;

    if (multinoteCount == 0 || (multinoteCount == drumMultinoteCount && IGNORE_DRUM)) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    std::vector<std::pair<Shape, unsigned int>> shapeScoreFreq;
    std::vector<std::pair<Shape, double>> shapeScoreWPlike;
    double neighborUpdatingTime, findBestShapeTime, mergeTime;
    std::cout << "Iter, Avg neighbor number, Found shapes count, Shape, Score, Multinote count, Iteration time, Neighbor update time, Find best shape time, Merge time" << std::endl;
    for (int iterCount = 0; iterCount < bpeIter; ++iterCount) {
        if (!verbose && iterCount != 0) 
            std::cout << "\33[2K\r"; // "\33[2K" is VT100 escape code that clear entire line
        std::cout << iterCount;
        std::chrono::time_point<std::chrono::system_clock>iterStartTimePoint = std::chrono::system_clock::now();
        std::chrono::time_point<std::chrono::system_clock>partStartTimePoint = std::chrono::system_clock::now();
        size_t totalNeighborNumber = updateNeighbor(corpus, shapeDict, nth, ignoreDrum); 
        neighborUpdatingTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;

        // clac shape scores
        Shape maxScoreShape;
        partStartTimePoint = std::chrono::system_clock::now();
        if (scoring == "default") {
            shapeScoring<unsigned int>(corpus, shapeDict, shapeScoreFreq, scoring, mergeCondition, samplingRate, ignoreDrum, verbose);
            std::pair<Shape, unsigned int> maxValPair = findMaxValPair(shapeScoreFreq);
            if (maxValPair.second < minScoreLimit) {
                std::cout << "\nEnd iterations because found best score < minScoreLimit\n";
                break;
            }
            maxScoreShape = maxValPair.first;
            std::cout << ", " << (double) totalNeighborNumber / multinoteCount << ", "
                      << shapeScoreFreq.size() << ", "
                      << "\"" << shape2str(maxScoreShape) << "\", "
                      << maxValPair.second << ", ";
            shapeScoreFreq.clear();
        }
        else {
            shapeScoring<double>(corpus, shapeDict, shapeScoreWPlike, scoring, mergeCondition, samplingRate, ignoreDrum, verbose);
            std::pair<Shape, double> maxValPair = findMaxValPair(shapeScoreWPlike);
            if (maxValPair.second < minScoreLimit) {
                std::cout << "\nEnd iterations because found best score < minScoreLimit\n";
                break;
            }
            maxScoreShape = maxValPair.first;
            std::cout << ", " << (double) totalNeighborNumber / multinoteCount << ", "
                      << shapeScoreWPlike.size() << ", "
                      << "\"" << shape2str(maxScoreShape) << "\", "
                      << maxValPair.second << ", ";
            shapeScoreWPlike.clear();
        }
        findBestShapeTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;

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
                // ignore drum?
                if (corpus.piecesTP[i][j] == 128 && IGNORE_DRUM) continue;
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
                        // if (shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()].size()
                        //         + shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()].size()
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
        
        multinoteCount = corpus.getMultiNoteCount();
        mergeTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;

        std::cout << multinoteCount << ", ";
        std::cout << (std::chrono::system_clock::now() - iterStartTimePoint) / onSencondDur << ", "
            << neighborUpdatingTime << ", "
            << findBestShapeTime << ", "
            << mergeTime;
        if (verbose) std::cout << std::endl;
        else         std::cout.flush();

        // corpus.shrink();
    }
    if (!verbose) std::cout << '\n';

    avgMulpi = calculateAvgMulpiSize(corpus, ignoreDrum);
    std::cout << "Ending multinote count: " << multinoteCount
        << ", Ending average mulpi: " << avgMulpi
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
    std::cout << "Writing done. Writing used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / onSencondDur << '\n'
        << "Total used time: " << (std::chrono::system_clock::now() - programStartTimePoint) / onSencondDur << std::endl;
    return 0;
}
