#include "classes.hpp"
#include "functions.hpp"
#include <string>
#include <algorithm>
#include <unistd.h>

int main(int argc, char *argv[]) {
    // read and validate args
    int cmd_opt = 0;
    bool clearLine = false;
    bool doLog = false;
    int nonOptStartIndex = 1;
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
                std::cout << "./apply_vocab [-log] [-clearLine] inCorpusDirPath outCorpusFilePath shapeVocabularyFilePath" << std::endl;
                return 1;
            default:
                std::cout << "./apply_vocab [-log] [-clearLine] inCorpusDirPath outCorpusFilePath shapeVocabularyFilePath" << std::endl;
                exit(1);
        }
    }
    if (argc - nonOptStartIndex != 3) {
        std::cout << "Bad number of non-optional arguments: " << argc - nonOptStartIndex << " != 3\n";
        std::cout << "./apply_vocab [-log] [-clearLine] inCorpusDirPath outCorpusFilePath shapeVocabularyFilePath" << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[nonOptStartIndex]);
    std::string outCorpusFilePath(argv[nonOptStartIndex+1]);
    std::string vocabFilePath(argv[nonOptStartIndex+2]);

    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusFilePath: " << outCorpusFilePath << '\n'
        << "vocabFilePath: " << vocabFilePath << std::endl;

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

    std::ifstream vocabFile(vocabFilePath, std::ios::in | std::ios::binary);
    if (!vocabFile.is_open()) {
        std::cout << "Failed to open vocab input file: " << vocabFilePath << std::endl;
        return 1;
    }
    std::cout << "Input shape vocab path: " << vocabFilePath << std::endl;

    std::cout << "Output merged corpus file: " << outCorpusFilePath << '\n'
        << "Reading input files" << std::endl;

    std::chrono::duration<double> oneSencondDur = std::chrono::duration<double>(1);
    std::chrono::time_point<std::chrono::system_clock> programStartTimePoint = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> ioStartTimePoint = std::chrono::system_clock::now();

    // read parameters
    std::map<std::string, std::string> paras = readParasFile(parasFile);
    int nth, maxDur, maxTrackNum;
    // stoi: c++11 thing
    nth = stoi(paras[std::string("nth")]);
    maxDur = stoi(paras[std::string("max_duration")]);
    maxTrackNum = stoi(paras[std::string("max_track_number")]);
    if (nth <= 0 || maxDur <= 0 || maxDur > 255 || maxTrackNum <= 0) {
        std::cout << "Corpus parameter error" << '\n'
                << "nth: " << nth << '\n'
                << "maxDuration: " << maxDur << '\n'
                << "maxTrackNum: " << maxTrackNum << std::endl;
        return 1;
    }

    // read notes from corpus
    Corpus corpus = readCorpusFile(inCorpusFile, nth);
    std::cout << "Reading done. There are " << corpus.piecesTP.size() << " pieces" << std::endl;

    std::vector<Shape> shapeDict = getDefaultShapeDict();

    // read shapes from vocab file
    vocabFile.seekg(0, std::ios::beg);
    std::string line;
    while (vocabFile.good()) {
        std::getline(vocabFile, line, '\n');
        if (line.size() == 0) continue;
        line.pop_back(); // because last character must be ';'
        for (int i = 0; i < line.size(); ++i) {
            if (line[i] == ',' || line[i] == ';') {
                line[i] = ' ';
            }
        }
        Shape shape;
        std::stringstream lineSS(line);
        while(lineSS.good()) {
            int isCont = 0, relOnset, relPitch, relDur;
            std::string relOnsetStr, relPitchStr, relDurStr;
            lineSS >> relOnsetStr >> relPitchStr >> relDurStr;
            if (relDurStr.back() == '~') {
                isCont = 1;
                relDurStr.pop_back();
            }
            relOnset = b36strtoi(relOnsetStr.c_str());
            relPitch = b36strtoi(relPitchStr.c_str());
            relDur = b36strtoi(relDurStr.c_str());
            shape.push_back(RelNote(isCont, relOnset, relPitch, relDur));
        }
        // std::cout << shape2str(shape) << '\n' << line << std::endl;
        shapeDict.push_back(shape);
        line.clear();
    }
    vocabFile.close();
    if (shapeDict.size() == 2) {
        std::cout << "Empty shape vocab file\n";
        return 0;
    }
    std::cout << "Shape vocab size: " << shapeDict.size() - 2 << "(+2)" << std::endl;

    // sort and count notes
    corpus.sortAllTracks();
    size_t startMultinoteCount, multinoteCount;
    startMultinoteCount = multinoteCount = corpus.getMultiNoteCount();
    double startAvgMulpi = calculateAvgMulpiSize(corpus, false);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start multinote count: " << multinoteCount
            << ", Start average mulpi: " << avgMulpi
            << ", Reading used time: " <<  (std::chrono::system_clock::now() - ioStartTimePoint) / oneSencondDur << std::endl;

    if (multinoteCount == 0) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    std::vector<std::pair<Shape, double>> shapeScoreWPlike;
    std::chrono::time_point<std::chrono::system_clock>iterStartTimePoint;
    std::chrono::time_point<std::chrono::system_clock>partStartTimePoint;
    double iterTime, mergeTime, metricsTime = 0.0;
    double shapeEntropy = 0.0, allEntropy = 0.0;
    if (doLog) {
        std::cout << "Index, Avg neighbor number, Shape, "
                << "Multinote count, Shape entropy, All attribute entropy, "
                << "Iteration time, Merge time" << std::endl;
    }
    // start from 2 because index 0, 1 are default shapes
    for (int shapeIndex = 2; shapeIndex < shapeDict.size(); ++shapeIndex) {
        iterStartTimePoint = std::chrono::system_clock::now();
        if (doLog) {
            if (clearLine && shapeIndex != 2) {
                std::cout << "\33[2K\r"; // "\33[2K" is VT100 escape code that clear entire line
            }
            std::cout << shapeIndex;
        }
        size_t totalNeighborNumber = updateNeighbor(corpus, shapeDict, nth);

        Shape mergingShape = shapeDict[shapeIndex];
        if (doLog){
            std::cout << ", " << (double) totalNeighborNumber / multinoteCount << ", "
                      << "\"" << shape2str(mergingShape) << "\", ";
        }

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
                        //         != mergingShape.size()) {
                        //     continue;
                        // }
                        Shape s = getShapeOfMultiNotePair(
                            corpus.piecesMN[i][j][k],
                            corpus.piecesMN[i][j][k+n],
                            shapeDict
                        );
                        if (s == mergingShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // the first relnote in the new shape is correspond to the first relnote in left multinote's original shape
                            uint8_t newUnit = shapeDict[corpus.piecesMN[i][j][k].shapeIndex][0].relDur * corpus.piecesMN[i][j][k].unit / mergingShape[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newUnit > maxDur) break;
                            corpus.piecesMN[i][j][k].unit = newUnit;
                            corpus.piecesMN[i][j][k].shapeIndex = shapeIndex;

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
            std::cout << iterTime << ", " << mergeTime;
            if (clearLine)  std::cout.flush();
            else            std::cout << std::endl;
        }
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
    std::cout << "Ending multinote count: " << multinoteCount
        << ", Ending average mulpi: " << avgMulpi
        << ", End shape entropy: " << shapeEntropy
        << ", End all attribute entropy: " << allEntropy
        << ", Multinote reduce rate: " << 1 - (double) multinoteCount / startMultinoteCount
        << ", Average mulpi reduce rate: " << 1 - avgMulpi / startAvgMulpi << std::endl;

    // Write files
    std::ofstream outCorpusFile(outCorpusFilePath, std::ios::out | std::ios::trunc);
    if (!outCorpusFile.is_open()) {
        std::cout << "Failed to open merged corpus output file: " << outCorpusFilePath << std::endl;
        return 1;
    }

    ioStartTimePoint = std::chrono::system_clock::now();
    std::cout << "Writing merged corpus file" << std::endl;
    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum);
    std::cout << "Writing done. Writing used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / oneSencondDur << '\n'
        << "Total used time: " << (std::chrono::system_clock::now() - programStartTimePoint) / oneSencondDur - metricsTime << std::endl;
    return 0;
}