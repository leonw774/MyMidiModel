#include "corpus.hpp"
#include "shapescore.hpp"
#include <string>
#include <algorithm>
#include <ctime>

int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 4 && argc != 5) {
        std::cout << "./learn_vocab inCorpusDirPath outCorpusFilePath shapeVocabularyFilePath (--verbose)" << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[1]);
    std::string outCorpusFilePath(argv[2]);
    std::string vocabFilePath(argv[3]);
    bool verbose = false;
    if (argc == 5) {
        std::string v(argv[4]);
        if (v != "--verbose") {
            std::cout << "./learn_vocab inCorpusDirPath outCorpusFilePath shapeVocabularyFilePath (--verbose)" << std::endl;
            return 1;
        }
        verbose = true;
    }
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

    std::ofstream outCorpusFile(outCorpusFilePath, std::ios::out | std::ios::trunc);
    if (!outCorpusFile.is_open()) {
        std::cout << "Failed to open merged corpus output file: " << outCorpusFilePath << std::endl;
        return 1;
    }
    std::cout << "Output merged corpus file: " << outCorpusFilePath << '\n'
        << "Reading input files" << std::endl;

    std::chrono::duration<double> onSencondDur = std::chrono::seconds(1);
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

    // read shapes from vocab file
    vocabFile.seekg(0, std::ios::beg);
    std::string line;
    while (vocabFile.good()) {
        std::getline(vocabFile, line, '\n');
        if (line[0] != 'S') {
            if (line.size() > 1) {
                std::cout << "shape vocab format error" << std::endl;
                return 1;
            }
            else {
                continue; // is just newline
            }
        }
        line.pop_back(); // because last character must be ';'
        for (int i = 0; i < line.size(); ++i) {
            if (line[i] == ',' || line[i] == ';') {
                line[i] = ' ';
            }
        }
        Shape shape;
        std::stringstream lineSS(line.substr(1));
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
    std::cout << "Shape vocab size: " << shapeDict.size() << std::endl;

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

    std::cout << "Start multinote count: " << multinoteCount
        << ", Drum's multinote count: " << drumMultinoteCount
        << ", Start average mulpi: " << avgMulpi
        << ", Reading used time: " <<  (std::chrono::system_clock::now() - ioStartTimePoint) / onSencondDur << std::endl;

    if (multinoteCount == 0 || multinoteCount == drumMultinoteCount) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    std::vector<std::pair<Shape, unsigned int>> shapeScoreFreq;
    std::vector<std::pair<Shape, double>> shapeScoreWPlike;
    double neighborUpdatingTime, shapeScoringTime, findMaxTime, mergeTime;
    std::cout << "Index, Shape, Score, Multinote count, Iteration time, Neighbor updating time, Merge time" << std::endl;
    // start from 2 because index 0, 1 are default shapes
    for (int shapeIndex = 2; shapeIndex < shapeDict.size(); ++shapeIndex) {
        if (!verbose && shapeIndex != 0) 
            std::cout << "\33[2K\r"; // "\33[2K" is VT100 escape code that clear entire line
        std::cout << shapeIndex;
        std::chrono::time_point<std::chrono::system_clock>iterStartTimePoint = std::chrono::system_clock::now();
        std::chrono::time_point<std::chrono::system_clock>partStartTimePoint = std::chrono::system_clock::now();
        updateNeighbor(corpus, shapeDict);
        neighborUpdatingTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;

        Shape mergingShape = shapeDict[shapeIndex];
        if (verbose) std::cout << ", \"" << shape2str(mergingShape) << "\", ";

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
                        if (corpus.piecesMN[i][j][k].vel != corpus.piecesMN[i][j][k+n].vel
                                || corpus.piecesMN[i][j][k].vel == 0
                                || corpus.piecesMN[i][j][k+n].vel == 0) {
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
                            shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()],
                            shapeDict[corpus.piecesMN[i][j][k+n].getShapeIndex()]
                        );
                        if (s == mergingShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // the first relnote in the new shape is correspond to the first relnote in left multinote's original shape
                            uint8_t newUnit = shapeDict[corpus.piecesMN[i][j][k].getShapeIndex()][0].relDur * corpus.piecesMN[i][j][k].unit / mergingShape[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newUnit > maxDur) break;
                            corpus.piecesMN[i][j][k].unit = newUnit;
                            corpus.piecesMN[i][j][k].setShapeIndex(shapeIndex);

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
        mergeTime = (std::chrono::system_clock::now() - partStartTimePoint) / onSencondDur;
        
        std::cout << multinoteCount << ", ";
        std::cout << (std::chrono::system_clock::now() - iterStartTimePoint) / onSencondDur << ", "
            << neighborUpdatingTime << ", "
            << shapeScoringTime << ", "
            << findMaxTime << ", "
            << mergeTime;
        if (verbose) std::cout << std::endl;
        else         std::cout.flush();
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
    std::cout << "Writing merged corpus file" << std::endl;
    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum, positionMethod);
    std::cout << "Writing done. Writing used time: " << (std::chrono::system_clock::now() - ioStartTimePoint) / onSencondDur << '\n'
        << "Total used time: " << (std::chrono::system_clock::now() - programStartTimePoint) / onSencondDur << std::endl;
    return 0;
}