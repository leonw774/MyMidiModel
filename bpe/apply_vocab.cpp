#include "classes.hpp"
#include "functions.hpp"
#include <string>
#include <algorithm>
#include <unistd.h>
#include "omp.h"

int main(int argc, char *argv[]) {
    // read and validate args
    int cmd_opt = 0;
    bool doLog = false;
    int nonOptStartIndex = 1;
    std::string cmdLineUsage =
        "./apply_vocab [-log] inCorpusDirPath outCorpusFilePath shapeVocabFilePath [workerNum]";
    while ((cmd_opt = getopt(argc, argv, "l:c:")) != -1) {
        nonOptStartIndex++;
        switch (cmd_opt) {
            case 'l':
                doLog = optarg;
                break;
            case '?':
                if (isprint(optopt)) {
                    std::cout << "Bad argument: " << argv[optopt] << "\n";
                }
                std::cout << cmdLineUsage<< std::endl;
                return 1;
            default:
                std::cout << cmdLineUsage << std::endl;
                exit(1);
        }
    }
    if ((argc - nonOptStartIndex != 3) && (argc - nonOptStartIndex != 4)) {
        std::cout << "Bad number of non-optional arguments: should be 3 or 4. Get "
            << argc - nonOptStartIndex << "\n";
        for (int i = 0; i < argc; ++i) {
            std::cout << argv[i] << " ";
        }
        std::cout << "\n" << cmdLineUsage << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[nonOptStartIndex]);
    std::string outCorpusFilePath(argv[nonOptStartIndex+1]);
    std::string vocabFilePath(argv[nonOptStartIndex+2]);
    int workerNum = -1; // -1 means use default
    if (argc - nonOptStartIndex == 4) {
        workerNum = atoi(argv[nonOptStartIndex+3]);
        omp_set_num_threads(workerNum);
    }

    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusFilePath: " << outCorpusFilePath << '\n'
        << "vocabFilePath: " << vocabFilePath << '\n'
        << "workerNum: " << workerNum << std::endl;

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

    std::chrono::duration<double> oneSecond = std::chrono::duration<double>(1);
    std::chrono::time_point<std::chrono::system_clock>
        programStartTime = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock>
        ioStartTime = std::chrono::system_clock::now();

    // read parameters
    std::map<std::string, std::string> paras = readParasFile(parasFile);
    int tpq, maxDur, maxTrackNum;
    // stoi: c++11 thing
    tpq = stoi(paras[std::string("tpq")]);
    maxDur = stoi(paras[std::string("max_duration")]);
    maxTrackNum = stoi(paras[std::string("max_track_number")]);
    if (tpq <= 0 || maxDur <= 0 || maxDur > 255 || maxTrackNum <= 0) {
        std::cout << "Corpus parameter error" << '\n'
                << "tpq: " << tpq << '\n'
                << "maxDuration: " << maxDur << '\n'
                << "maxTrackNum: " << maxTrackNum << std::endl;
        return 1;
    }

    // read shapes from vocab file
    std::vector<Shape> shapeDict = getDefaultShapeDict();
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
            shape.push_back(RelNote(relOnset, relPitch, relDur, isCont));
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

    // read notes from corpus
    Corpus corpus = readCorpusFile(inCorpusFile, tpq, maxTrackNum);
    int numTracks = 0;
    for (int i = 0; i < corpus.pieceNum; i++) {
        numTracks += corpus.trackInstrMaps[i].size();
    }
    std::cout << "Reading done. There are " << corpus.pieceNum
        << " pieces and "  << numTracks << " tracks." << std::endl;

    // sort and count notes
    corpus.sortAllTracks();
    size_t startMultinoteCount, multinoteCount;
    startMultinoteCount = multinoteCount = corpus.getMultiNoteCount();
    double startAvgMulpi = calculateAvgMulpiSize(corpus, false);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start multinote count: " << multinoteCount
        << ", Start average mulpi: " << avgMulpi
        << ", Reading used time: "
        <<  (std::chrono::system_clock::now() - ioStartTime) / oneSecond << std::endl;

    if (multinoteCount == 0) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    std::chrono::time_point<std::chrono::system_clock> iterStartTime;
    std::chrono::time_point<std::chrono::system_clock> partStartTime;
    double iterTime, mergeTime, metricsTime = 0.0;
    double totalMergeTime = 0.0;
    if (doLog) {
        std::cout << "Index, Avg neighbor number, Shape, "
            "Multinote count, Iteration time, Merge time" << std::endl;
    }
    // start from 2 because index 0, 1 are default shapes
    for (int shapeIndex = 2; shapeIndex < shapeDict.size(); ++shapeIndex) {
        iterStartTime = std::chrono::system_clock::now();
        size_t totalNeighborNumber = updateNeighbor(corpus, shapeDict, tpq);
        Shape mergingShape = shapeDict[shapeIndex];
        if (doLog){
            std::cout << shapeIndex
                << ", " << (double) totalNeighborNumber / multinoteCount
                << ", \"" << shape2str(mergingShape) << "\", ";
        }

        partStartTime = std::chrono::system_clock::now();
        // merge MultiNotes with newly added shape
        // for each piece
        #pragma omp parallel for
        for (int i = 0; i < corpus.pieceNum; ++i) {
            // for each track
            std::vector<Track>& piece = corpus.mns[i];
            #pragma omp parallel for
            for (int j = 0; j < piece.size(); ++j) {
                Track &track = piece[j];
                // for each multinote
                for (int k = 0; k < track.size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n < track[k].neighbor; ++n) {
                        if (k + n >= track.size()) {
                            continue;
                        }
                        if (track[k].vel == 0
                            || track[k+n].vel == 0
                            || track[k].vel != track[k+n].vel) {
                            continue;
                        }
                        Shape s = getShapeOfMultiNotePair(track[k], track[k+n], shapeDict);
                        if (s == mergingShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // the first relnote in the new shape is correspond to
                            // the first relnote in left multinote's original shape
                            uint8_t newStretch =
                                shapeDict[track[k].shapeIndex][0].relDur
                                * track[k].stretch / mergingShape[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newStretch > maxDur) continue;
                            track[k].stretch = newStretch;
                            track[k].shapeIndex = shapeIndex;

                            // mark right multinote to be removed by have vel set to 0
                            track[k+n].vel = 0;
                            break;
                        }
                    }
                }
                // remove multinotes with vel == 0
                track.erase(
                    std::remove_if(track.begin(), track.end(), [] (const MultiNote& m) {
                        return m.vel == 0;
                    }),
                    track.end()
                );
            }
        }
        mergeTime = (std::chrono::system_clock::now() - partStartTime) / oneSecond;
        totalMergeTime += mergeTime;
        iterTime = (std::chrono::system_clock::now() - iterStartTime) / oneSecond;
        if (doLog) {
            partStartTime = std::chrono::system_clock::now();
            multinoteCount = corpus.getMultiNoteCount();
            // To exclude the time used on calculating metrics
            metricsTime += (std::chrono::system_clock::now() - partStartTime) / oneSecond;
            std::cout << multinoteCount
                << ", " << iterTime
                << ", " << mergeTime;
            std::cout << std::endl;
        }
    }

    if (!doLog) {
        multinoteCount = corpus.getMultiNoteCount();
    }
    avgMulpi = calculateAvgMulpiSize(corpus);
    std::cout << "Ending multinote count: " << multinoteCount
        << ", Ending average mulpi: " << avgMulpi
        << ", Total merge time: " << totalMergeTime
        << std::endl;

    // Write files
    std::ofstream outCorpusFile(outCorpusFilePath, std::ios::out | std::ios::trunc);
    if (!outCorpusFile.is_open()) {
        std::cout << "Failed to open merged corpus output file: "
            << outCorpusFilePath << std::endl;
        return 1;
    }

    ioStartTime = std::chrono::system_clock::now();
    std::cout << "Writing merged corpus file" << std::endl;
    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum);
    std::cout << "Writing done. Writing used time: "
        << (std::chrono::system_clock::now() - ioStartTime) / oneSecond << '\n'
        << "Total used time: "
        << (std::chrono::system_clock::now() - programStartTime) / oneSecond - metricsTime
        << std::endl;
    return 0;
}
